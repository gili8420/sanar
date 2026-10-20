#!/usr/bin/env python3
"""Who spoke when, once per recording: pyannote community-1 over the stitched audio.

Writes speakers.json: exclusive turns (one speaker at a time; transcribe.py chunks the audio
and labels words by them), overlapping-speech spans, per-speaker totals and, with --voices,
the closest enrolled voice for each speaker.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# Ungated mirror of pyannote/speaker-diarization-community-1 (CC-BY-4.0): no HF token needed.
PIPELINE = "pyannote-community/speaker-diarization-community-1"
SR = 16000
AUDIO_EXT = (".wav", ".m4a", ".mp3", ".ogg", ".opus", ".flac", ".aac", ".aiff", ".mp4", ".webm")

# ponytail: fixed cosine bands for the pipeline's WeSpeaker ResNet34 embeddings. If similar
# voices get swapped, raise MATCH; calibrate on recordings of people you know.
MATCH, REVIEW = 0.5, 0.35


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
         "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def speaker_bounds(spec: str) -> dict:
    """'3' -> exactly three speakers, '2-5' -> between two and five."""
    if not spec:
        return {}
    lo, _, hi = spec.partition("-")
    return {"min_speakers": int(lo), "max_speakers": int(hi)} if hi else {"num_speakers": int(lo)}


def match_voices(pipe, centroids: np.ndarray, voices_dir: str) -> dict[int, dict]:
    """Greedy one-to-one match of speaker centroids to enrolled voice clips (name = file stem)."""
    import torch

    files = sorted(f for f in os.listdir(voices_dir) if f.lower().endswith(AUDIO_EXT))
    if not files or not len(centroids):
        return {}
    names = [os.path.splitext(f)[0] for f in files]
    enrolled = np.vstack([
        pipe._embedding(torch.from_numpy(decode(os.path.join(voices_dir, f)))[None, None])
        for f in files
    ])
    norm = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)  # noqa: E731
    sim = norm(centroids) @ norm(enrolled).T

    out, used = {}, set()
    for s, v in sorted(np.ndindex(sim.shape), key=lambda ij: -sim[ij]):
        if s in out or v in used or sim[s, v] < REVIEW:
            continue
        out[s] = {"name": names[v], "similarity": round(float(sim[s, v]), 3),
                  "band": "matched" if sim[s, v] >= MATCH else "review"}
        used.add(v)
    for s in range(len(centroids)):  # unmatched: still say who was closest
        if s not in out:
            v = int(np.argmax(sim[s]))
            out[s] = {"name": None, "closest": names[v],
                      "similarity": round(float(sim[s, v]), 3), "band": "unmatched"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True, help="speakers.json")
    ap.add_argument("--speakers", default="", help="exact count '3' or range '2-5'")
    ap.add_argument("--voices", default="", help="dir of <Name>.<audio> clips, 10-30 s each")
    args = ap.parse_args()

    import torch
    from pyannote.audio import Pipeline

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.monotonic()
    pipe = Pipeline.from_pretrained(PIPELINE)
    pipe.to(torch.device(device))
    audio = decode(args.audio)
    t1 = time.monotonic()
    out = pipe({"waveform": torch.from_numpy(audio)[None], "sample_rate": SR},
               **speaker_bounds(args.speakers))
    wall = time.monotonic() - t1

    labels = out.speaker_diarization.labels()  # row order of out.speaker_embeddings
    turns = sorted((round(t.start, 3), round(t.end, 3), lab) for t, _, lab
                   in out.exclusive_speaker_diarization.itertracks(yield_label=True))
    first = {}
    for s, _, lab in turns:
        first.setdefault(lab, s)
    order = sorted(labels, key=lambda lab: first.get(lab, float("inf")))
    sid = {lab: f"S{i + 1}" for i, lab in enumerate(order)}

    voices = {}
    if args.voices and out.speaker_embeddings is not None:
        voices = match_voices(pipe, out.speaker_embeddings, args.voices)

    speakers = []
    for lab in order:
        mine = [e - s for s, e, x in turns if x == lab]
        speakers.append({"id": sid[lab], "seconds": round(sum(mine), 1), "turns": len(mine),
                         "voice": voices.get(labels.index(lab))})
    overlaps = [[round(seg.start, 3), round(seg.end, 3),
                 sorted(sid[x] for x in out.speaker_diarization.crop(seg).labels())]
                for seg in out.speaker_diarization.get_overlap()]

    doc = {
        "audio": os.path.abspath(args.audio),
        "pipeline": PIPELINE,
        "device": device,
        "audio_seconds": round(len(audio) / SR, 2),
        "load_seconds": round(t1 - t0, 2),
        "wall_seconds": round(wall, 2),
        "realtime_factor": round(len(audio) / SR / wall, 1) if wall > 0 else None,
        "speakers": speakers,
        "turns": [[s, e, sid[lab]] for s, e, lab in turns],
        "overlaps": overlaps,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)

    for k in ("device", "audio_seconds", "load_seconds", "wall_seconds", "realtime_factor"):
        print(f"{k}={doc[k]}")
    print(f"speakers={len(speakers)} turns={len(turns)} overlaps={len(overlaps)}")
    for s in speakers:
        v = s["voice"] or {}
        who = v.get("name") or f"closest {v.get('closest')}"
        print(f"  {s['id']}: {s['seconds']}s in {s['turns']} turns"
              + (f", voice {who} sim={v['similarity']} ({v['band']})" if v else ""))
    print(f"out={args.out}")
    if not turns:
        print("warn: no speech found", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
