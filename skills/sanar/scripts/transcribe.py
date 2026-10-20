#!/usr/bin/env python3
"""Whisper large-v3 passes over one recording, one per language, primed with a ranked glossary.

The audio is cut into chunks of at most 28 s along the diarized speaker turns (speakers.json).
Every chunk gets the whole glossary as its prompt, every pass sees the same chunks, and every
word is labelled with the speaker whose turn contains it.

Backends: mlx-whisper (Apple GPU: one encoder pass per chunk shared by all languages, batched
greedy decoding) or faster-whisper (CUDA / CPU: BatchedInferencePipeline).
Writes pass-<lang>.txt (`@cNNN` chunk headers, `[hh:mm:ss] S1: text` lines) and
pass-<lang>.txt.json (run metadata) to --out-dir.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter

import numpy as np

# Whisper's prompt window is n_text_ctx // 2 - 1 == 223 tokens. Leave headroom for the
# special tokens the decoder prepends.
PROMPT_TOKEN_BUDGET = 200

TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SR = 16000
MAX_CHUNK = 28.0  # s; under the 30 s window, with room for PAD on both sides
MAX_GAP = 2.0  # s; a longer pause starts a new chunk, silence inside a window invites hallucination
PAD = 0.15  # s of context around each chunk
# Whisper's own thresholds: a chunk failing them is redone with temperature fallback.
COMPRESSION_MAX, LOGPROB_MIN, NO_SPEECH_MIN = 2.4, -1.0, 0.6

MLX_REPOS = {"large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
             "turbo": "mlx-community/whisper-large-v3-turbo"}
# OpenAI's word-alignment heads (decoder layer, head). Without them mlx-whisper averages every
# head of the upper decoder half, which blurs word timings at speaker changes.
ALIGNMENT_HEADS = {
    "large-v3-turbo": [[2, 4], [2, 11], [3, 3], [3, 6], [3, 11], [3, 14]],
    "large-v3": [[7, 0], [10, 17], [12, 18], [13, 12], [16, 1], [17, 14], [19, 11], [21, 4],
                 [24, 1], [25, 6]],
}


def fmt_ts(sec: float) -> str:
    h, rem = divmod(int(max(0.0, sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_terms(path: str | None) -> list[str]:
    if not path:
        return []
    out, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#") or t.lower() in seen:
                continue
            seen.add(t.lower())
            out.append(t)
    return out


def approx_len(text: str) -> int:
    """Fallback token estimate: Whisper's multilingual BPE runs ~2.5 chars/token on ru+en."""
    return int(len(text) / 2.5) + 1


def build_prompt(terms: list[str], count_tokens) -> tuple[str | None, int, int]:
    """Pack a greedy prefix of the ranked glossary into the prompt window.

    ponytail: O(n^2) re-encoding, ~ms for a few thousand terms. Batch it if that changes.
    """
    if not terms:
        return None, 0, 0
    kept: list[str] = []
    for t in terms:
        if count_tokens(", ".join(kept + [t]) + ".") > PROMPT_TOKEN_BUDGET:
            break
        kept.append(t)
    if not kept:
        return None, 0, len(terms)
    return ", ".join(kept) + ".", len(kept), len(terms) - len(kept)


def load_audio(path: str) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
         "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def pack(turns: list[list]) -> list[list[float]]:
    """Greedily join consecutive speaker turns into chunks of at most MAX_CHUNK seconds."""
    chunks: list[list[float]] = []
    for s, e, _ in turns:
        # ponytail: a turn longer than MAX_CHUNK is cut blindly into equal parts and may clip
        # a word at each cut; cut at the quietest frame instead if that shows up in merges.
        n = max(1, math.ceil((e - s) / MAX_CHUNK))
        step = (e - s) / n
        for k in range(n - 1):
            chunks.append([s + k * step, s + (k + 1) * step])
        s += (n - 1) * step
        if chunks and e - chunks[-1][0] <= MAX_CHUNK and s - chunks[-1][1] <= MAX_GAP:
            chunks[-1][1] = max(chunks[-1][1], e)
        else:
            chunks.append([s, e])
    return chunks


def speaker_at(turns: list[list], starts: list[float], t: float) -> str:
    """Speaker whose turn contains t, else the one whose turn edge is nearest."""
    i = max(bisect.bisect_right(starts, t) - 1, 0)
    if t <= turns[i][1] or i + 1 == len(turns):
        return turns[i][2]
    return turns[i][2] if t - turns[i][1] < turns[i + 1][0] - t else turns[i + 1][2]


def to_lines(words, turns, starts) -> list[str]:
    lines: list[list] = []
    for s, e, w in words:
        spk = speaker_at(turns, starts, (s + e) / 2)
        if lines and lines[-1][1] == spk:
            lines[-1][2].append(w)
        else:
            lines.append([s, spk, [w]])
    return [f"[{fmt_ts(s)}] {spk}: {''.join(ws).strip()}" for s, spk, ws in lines]


def padded(chunks: list[list[float]], total: float) -> list[list[float]]:
    """Chunk spans widened by PAD on each side, never into a neighbouring chunk."""
    return [[max(s - PAD, chunks[i - 1][1] if i else 0.0),
             min(e + PAD, chunks[i + 1][0] if i + 1 < len(chunks) else total)]
            for i, (s, e) in enumerate(chunks)]


def clip_of(audio: np.ndarray, span: list[float]) -> tuple[float, np.ndarray]:
    return span[0], audio[int(span[0] * SR): int(span[1] * SR)]


def verdict(text: str, compression: float, logprob: float, no_speech: float,
            truncated: bool, prompt: str | None) -> str | None:
    """None = keep; 'silence'/'prompt-echo' = drop; anything else = redo with fallback."""
    if no_speech > NO_SPEECH_MIN and logprob < LOGPROB_MIN:
        return "silence"
    if prompt and len(text) > 12 and text.strip(" .,") in prompt:
        return "prompt-echo"  # Whisper reciting the glossary over near-silence
    if truncated:
        return "truncated"
    if compression > COMPRESSION_MAX:
        return "compression"
    if logprob < LOGPROB_MIN:
        return "logprob"
    return None


def preload_cuda_libs() -> None:
    """Put pip-installed cuBLAS/cuDNN on the loader path for CTranslate2.

    ponytail: RTLD_GLOBAL preload beats exporting LD_LIBRARY_PATH and re-exec'ing.
    """
    import ctypes
    import glob
    import importlib

    sos = []
    for mod in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            d = os.path.dirname(importlib.import_module(mod).__file__)
        except Exception:
            continue
        sos += sorted(glob.glob(os.path.join(d, "lib*.so*")))
    for _ in range(2):  # second pass resolves inter-library load order
        for so in sos:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def pick_backend(explicit: str) -> str:
    if explicit:
        return explicit
    env = os.environ.get("SANAR_BACKEND", "")
    if env:
        return env
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "faster"


def run_mlx(args, langs, chunks, audio, terms, run):
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import DecodingOptions, decode
    from mlx_whisper.timing import find_alignment
    from mlx_whisper.tokenizer import get_tokenizer
    from mlx_whisper.transcribe import ModelHolder, transcribe

    name = args.model or os.environ.get("SANAR_MODEL") or "large-v3"
    repo = name if "/" in name else MLX_REPOS.get(name, f"mlx-community/whisper-{name}-mlx")
    t0 = time.monotonic()
    model = ModelHolder.get_model(repo, mx.float16)  # shared with the fallback transcribe()
    heads = next((h for k, h in ALIGNMENT_HEADS.items() if k in repo), None)
    if heads and max(l for l, _ in heads) < model.dims.n_text_layer:  # not e.g. distil-large-v3
        model.set_alignment_heads(np.array(heads))
    run.update(model=repo, device="metal", load_seconds=round(time.monotonic() - t0, 2))

    toks: dict = {}

    def tokenizer(lang):
        if lang not in toks:
            toks[lang] = get_tokenizer(True, num_languages=model.num_languages,
                                       language=lang, task="transcribe")
        return toks[lang]

    tok0 = tokenizer(None if langs[0] == "auto" else langs[0])
    prompt, used, dropped = build_prompt(terms, lambda t: len(tok0.encode(" " + t)))
    run.update(prompt=prompt, prompt_terms_used=used, prompt_terms_dropped=dropped)

    class Cached:  # find_alignment() re-runs the encoder on its input; hand it our features
        alignment_heads = model.alignment_heads

        @staticmethod
        def forward_with_cross_qk(feats, tokens):
            logits, _, qk = model.decoder(tokens, feats)
            return logits, qk

    limit = model.dims.n_text_ctx // 2 - 1
    res = {lang: [None] * len(chunks) for lang in langs}
    for b in range(0, len(chunks), args.batch):
        idx = range(b, min(b + args.batch, len(chunks)))
        clips = [clip_of(audio, chunks[i]) for i in idx]
        mels = mx.stack([pad_or_trim(log_mel_spectrogram(c, n_mels=model.dims.n_mels,
                                                         padding=N_SAMPLES), N_FRAMES, axis=-2)
                         for _, c in clips]).astype(mx.float16)
        t = time.monotonic()
        feats = model.encoder(mels)
        mx.eval(feats)
        run["encoder_seconds"] += time.monotonic() - t
        for lang in langs:
            t = time.monotonic()
            # Timestamp tokens anchor the decoder: without them it skipped whole utterances in
            # mixed-language chunks (English recall 92% -> 97% on a scripted ru/en meeting).
            opts = DecodingOptions(language=None if lang == "auto" else lang, prompt=prompt,
                                   without_timestamps=False, fp16=True)
            for k, (i, (off, clip), r) in enumerate(zip(idx, clips, decode(model, feats, opts))):
                why = verdict(r.text, r.compression_ratio, r.avg_logprob, r.no_speech_prob,
                              len(r.tokens) >= limit, prompt)
                if why is None:
                    tok = tokenizer(r.language)
                    timings = find_alignment(Cached, tok, [x for x in r.tokens if x < tok.eot],
                                             feats[k], int(len(clip) / SR * 100))
                    words = [(off + w.start, off + w.end, w.word) for w in timings]
                elif why in ("silence", "prompt-echo"):
                    words = []
                    run["dropped"][lang].append({"chunk": f"c{i + 1:03d}", "reason": why,
                                                 "text": r.text})
                else:
                    rr = transcribe(clip, path_or_hf_repo=repo, language=opts.language,
                                    initial_prompt=prompt, temperature=TEMPERATURES,
                                    condition_on_previous_text=False, word_timestamps=True,
                                    verbose=None)
                    words = [(off + w["start"], off + w["end"], w["word"])
                             for seg in rr["segments"] for w in seg.get("words", [])]
                    run["fallback"][lang].append({"chunk": f"c{i + 1:03d}", "reason": why})
                run["languages"][lang][r.language] += 1
                res[lang][i] = words
            run["decode_seconds"][lang] += time.monotonic() - t
    return res


def run_faster(args, langs, chunks, audio, terms, run):
    preload_cuda_libs()
    import ctranslate2
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from faster_whisper.transcribe import get_compression_ratio

    device = args.device or os.environ.get("SANAR_DEVICE") or ""
    if not device or device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute = (args.compute or os.environ.get("SANAR_COMPUTE")
               or ("float16" if device == "cuda" else "int8"))
    name = args.model or os.environ.get("SANAR_MODEL") or "large-v3"
    t0 = time.monotonic()
    m = WhisperModel(name, device=device, compute_type=compute)
    run.update(model=name, device=f"{device}/{compute}", load_seconds=round(time.monotonic() - t0, 2))

    count = approx_len
    hf = getattr(m, "hf_tokenizer", None)
    if hf is not None:
        count = lambda t: len(hf.encode(t).ids)  # noqa: E731
    prompt, used, dropped = build_prompt(terms, count)
    run.update(prompt=prompt, prompt_terms_used=used, prompt_terms_dropped=dropped)

    # ponytail: the encoder runs once per language here; sharing it like run_mlx needs
    # ctranslate2's low-level encode/generate. Do it if CUDA users run many languages.
    pipe = BatchedInferencePipeline(m)
    clips = [clip_of(audio, c) for c in chunks]
    starts = [off for off, _ in clips]
    res = {lang: [[] for _ in chunks] for lang in langs}
    for lang in langs:
        t = time.monotonic()
        language = None if lang == "auto" else lang
        segs, info = pipe.transcribe(
            audio, language=language, initial_prompt=prompt, batch_size=args.batch,
            clip_timestamps=[{"start": off, "end": off + len(c) / SR} for off, c in clips],
            word_timestamps=True, without_timestamps=False,  # see run_mlx
        )
        per = [[] for _ in chunks]
        for s in segs:
            per[max(bisect.bisect_right(starts, s.start + 1e-3) - 1, 0)].append(s)
        for i, ss in enumerate(per):
            if not ss:
                continue
            # batched mode stamps every segment of a chunk with the chunk's logprob/no_speech
            text = "".join(s.text for s in ss)
            why = verdict(text, get_compression_ratio(text), ss[0].avg_logprob,
                          ss[0].no_speech_prob, False, prompt)
            if why is None:
                res[lang][i] = [(w.start, w.end, w.word) for s in ss for w in s.words or []]
            elif why in ("silence", "prompt-echo"):
                run["dropped"][lang].append({"chunk": f"c{i + 1:03d}", "reason": why,
                                             "text": text})
            else:
                off, clip = clips[i]
                redo, _ = m.transcribe(clip, language=language, initial_prompt=prompt,
                                       temperature=list(TEMPERATURES), vad_filter=False,
                                       condition_on_previous_text=False, word_timestamps=True)
                res[lang][i] = [(off + w.start, off + w.end, w.word)
                                for r in redo for w in r.words or []]
                run["fallback"][lang].append({"chunk": f"c{i + 1:03d}", "reason": why})
        run["languages"][lang][info.language] += len(chunks)  # batched: first chunk's language
        run["decode_seconds"][lang] += time.monotonic() - t
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--speakers", required=True, help="speakers.json from diarize.py")
    ap.add_argument("--lang", required=True, help="comma-separated: ru,en | auto | any Whisper code")
    ap.add_argument("--terms", help="ranked glossary, one term per line, best-first")
    ap.add_argument("--out-dir", required=True, help="writes pass-<lang>.txt and .txt.json here")
    ap.add_argument("--backend", default="", choices=("", "mlx", "faster"))
    ap.add_argument("--device", default="", help="faster-whisper only: cuda | cpu | auto")
    ap.add_argument("--compute", default="", help="faster-whisper only: float16 | int8 | ...")
    ap.add_argument("--model", default="")
    # Bigger batches were faster all the way up on an M1 Pro (large-v3, 18 chunks: batch 4
    # 101 s, 8 77 s, 18 56 s); each chunk holds ~250 MB of cross-attention cache on large-v3.
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    ap.add_argument("--batch", type=int,
                    default=int(os.environ.get("SANAR_BATCH") or (16 if ram_gb >= 15 else 8)),
                    help="chunks decoded together; default 16 with >=16 GB RAM, else 8")
    args = ap.parse_args()

    for p in (args.audio, args.speakers):
        if not os.path.isfile(p):
            print(f"no such file: {p}", file=sys.stderr)
            return 2
    langs = [x.strip() for x in args.lang.split(",") if x.strip()]
    with open(args.speakers, encoding="utf-8") as f:
        turns = json.load(f)["turns"]
    if not turns:
        print("speakers.json has no speech turns; nothing to transcribe", file=sys.stderr)
        return 3

    backend = pick_backend(args.backend)
    audio = load_audio(args.audio)
    chunks = pack(turns)
    run = {"encoder_seconds": 0.0, "decode_seconds": Counter(), "fallback": {l: [] for l in langs},
           "dropped": {l: [] for l in langs}, "languages": {l: Counter() for l in langs}}
    t0 = time.monotonic()
    spans = padded(chunks, len(audio) / SR)
    res = (run_mlx if backend == "mlx" else run_faster)(args, langs, spans, audio,
                                                        load_terms(args.terms), run)
    wall = time.monotonic() - t0 - run["load_seconds"]

    starts = [t[0] for t in turns]
    audio_seconds = len(audio) / SR
    os.makedirs(args.out_dir, exist_ok=True)
    for lang in langs:
        out = os.path.join(args.out_dir, f"pass-{lang}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# sanar pass lang={lang} backend={backend} model={run['model']} "
                    f"chunks={len(chunks)}\n")
            for i, (chunk, words) in enumerate(zip(chunks, res[lang])):
                f.write(f"@c{i + 1:03d} {fmt_ts(chunk[0])}-{fmt_ts(chunk[1])}\n")
                for line in to_lines(words, turns, starts):
                    f.write(line + "\n")
        meta = {
            "audio": os.path.abspath(args.audio),
            "lang_requested": lang,
            "backend": backend,
            "model": run["model"],
            "device": run["device"],
            "chunks": len(chunks),
            "audio_seconds": round(audio_seconds, 2),
            "speech_seconds": round(sum(e - s for s, e in chunks), 2),
            "load_seconds": run["load_seconds"],
            "encoder_seconds": round(run["encoder_seconds"], 2),  # shared by all languages
            "decode_seconds": round(run["decode_seconds"][lang], 2),
            "wall_seconds": round(wall, 2),  # all languages together, model load excluded
            "realtime_factor": round(audio_seconds / wall, 2) if wall > 0 else None,
            "detected_languages": dict(run["languages"][lang]),
            "prompt_terms_total": len(load_terms(args.terms)),
            "prompt_terms_used": run["prompt_terms_used"],
            "prompt_terms_dropped": run["prompt_terms_dropped"],
            "prompt": run["prompt"],
            "fallback": run["fallback"][lang],
            "dropped": run["dropped"][lang],
        }
        with open(out + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"lang={lang} out={out} decode_seconds={meta['decode_seconds']} "
              f"fallback={len(meta['fallback'])} dropped={len(meta['dropped'])} "
              f"languages={meta['detected_languages']}")
    print(f"backend={backend} model={run['model']} device={run['device']} "
          f"load_seconds={run['load_seconds']}")
    print(f"chunks={len(chunks)} audio_seconds={round(audio_seconds, 2)} "
          f"encoder_seconds={round(run['encoder_seconds'], 2)} wall_seconds={round(wall, 2)} "
          f"realtime_factor={round(audio_seconds / wall, 2) if wall > 0 else None}")
    print(f"prompt_terms_used={run['prompt_terms_used']} "
          f"prompt_terms_dropped={run['prompt_terms_dropped']}")
    if run["prompt_terms_dropped"]:
        print(f"note: {run['prompt_terms_dropped']} glossary terms did not fit the "
              f"{PROMPT_TOKEN_BUDGET}-token prompt window; they must be handled in the merge "
              f"phase", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
