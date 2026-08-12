#!/usr/bin/env python3
"""One Whisper large-v3 pass over one audio file, primed with a ranked glossary.

Backends: mlx-whisper (Apple GPU) or faster-whisper (CUDA / CPU).
Writes `[hh:mm:ss] text` segments to --out and run metadata to <out>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

# Whisper's prompt window is n_text_ctx // 2 - 1 == 223 tokens. Leave headroom for the
# special tokens the decoder prepends.
PROMPT_TOKEN_BUDGET = 200

TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


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


def run_mlx(args, terms):
    import mlx_whisper

    model = args.model or os.environ.get("SANAR_MODEL") or "large-v3"
    if "/" not in model:
        model = f"mlx-community/whisper-{model}-mlx"

    count = approx_len
    try:
        from mlx_whisper.tokenizer import get_tokenizer

        tok = get_tokenizer(
            multilingual=True,
            language=None if args.lang == "auto" else args.lang,
            task="transcribe",
        )
        count = lambda t: len(tok.encode(t))  # noqa: E731
    except Exception as e:
        print(f"warn: mlx tokenizer unavailable ({e}), using char estimate", file=sys.stderr)

    prompt, used, dropped = build_prompt(terms, count)
    r = mlx_whisper.transcribe(
        args.audio,
        path_or_hf_repo=model,
        language=None if args.lang == "auto" else args.lang,
        task="transcribe",
        initial_prompt=prompt,
        condition_on_previous_text=not args.no_condition,
        temperature=TEMPERATURES,
        compression_ratio_threshold=2.4,
        verbose=False,
    )
    segs = [(s["start"], s["end"], s["text"].strip()) for s in r["segments"]]
    meta = {
        "model": model,
        "device": "metal",
        "detected_language": r.get("language"),
        "vad": False,
    }
    return segs, meta, prompt, used, dropped


def run_faster(args, terms):
    preload_cuda_libs()
    import ctranslate2
    from faster_whisper import WhisperModel

    device = args.device or os.environ.get("SANAR_DEVICE") or ""
    if not device or device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute = (
        args.compute
        or os.environ.get("SANAR_COMPUTE")
        or ("float16" if device == "cuda" else "int8")
    )
    model = args.model or os.environ.get("SANAR_MODEL") or "large-v3"

    m = WhisperModel(model, device=device, compute_type=compute)

    count = approx_len
    hf = getattr(m, "hf_tokenizer", None)
    if hf is not None:
        try:
            hf.encode("probe")
            count = lambda t: len(hf.encode(t).ids)  # noqa: E731
        except Exception:
            pass

    prompt, used, dropped = build_prompt(terms, count)
    segments, info = m.transcribe(
        args.audio,
        language=None if args.lang == "auto" else args.lang,
        task="transcribe",
        initial_prompt=prompt,
        condition_on_previous_text=not args.no_condition,
        temperature=list(TEMPERATURES),
        beam_size=5,
        vad_filter=True,  # cheap insurance against hallucination over silence
        word_timestamps=False,
    )
    segs = [(s.start, s.end, s.text.strip()) for s in segments]  # generator: consumes here
    meta = {
        "model": model,
        "device": f"{device}/{compute}",
        "detected_language": info.language,
        "language_probability": round(float(info.language_probability or 0), 4),
        "vad": True,
    }
    return segs, meta, prompt, used, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lang", required=True, help="ru | en | auto | any Whisper code")
    ap.add_argument("--terms", help="ranked glossary, one term per line, best-first")
    ap.add_argument("--out", required=True, help="output .txt (metadata goes to <out>.json)")
    ap.add_argument("--backend", default="", choices=("", "mlx", "faster"))
    ap.add_argument("--device", default="", help="faster-whisper only: cuda | cpu | auto")
    ap.add_argument("--compute", default="", help="faster-whisper only: float16 | int8 | ...")
    ap.add_argument("--model", default="")
    ap.add_argument(
        "--no-condition",
        action="store_true",
        help="stop conditioning on previous text; use after a decoder loop",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.audio):
        print(f"no such audio: {args.audio}", file=sys.stderr)
        return 2

    backend = pick_backend(args.backend)
    terms = load_terms(args.terms)
    t0 = time.monotonic()
    segs, meta, prompt, used, dropped = (run_mlx if backend == "mlx" else run_faster)(args, terms)
    wall = time.monotonic() - t0

    audio_end = segs[-1][1] if segs else 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# sanar pass lang={args.lang} backend={backend} model={meta['model']}\n")
        for st, _en, txt in segs:
            if txt:
                f.write(f"[{fmt_ts(st)}] {txt}\n")

    meta = {
        "audio": os.path.abspath(args.audio),
        "lang_requested": args.lang,
        "backend": backend,
        "segments": len(segs),
        "audio_seconds": round(audio_end, 2),
        "wall_seconds": round(wall, 2),
        "realtime_factor": round(audio_end / wall, 2) if wall > 0 else None,
        "prompt_terms_total": len(terms),
        "prompt_terms_used": used,
        "prompt_terms_dropped": dropped,
        "prompt": prompt,
        "condition_on_previous_text": not args.no_condition,
        **meta,
    }
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    for k in ("backend", "device", "model", "lang_requested", "detected_language",
              "segments", "audio_seconds", "wall_seconds", "realtime_factor",
              "prompt_terms_used", "prompt_terms_dropped"):
        print(f"{k}={meta.get(k)}")
    print(f"out={args.out}")
    if dropped:
        print(f"note: {dropped} glossary terms did not fit the {PROMPT_TOKEN_BUDGET}-token "
              f"prompt window; they must be handled in the merge phase", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
