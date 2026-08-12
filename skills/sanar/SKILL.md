---
name: sanar
description: Context-aware local transcription with Whisper large-v3. Use when the user runs /sanar, or asks to transcribe audio/video locally, run multi-language Whisper passes, or prime a transcription with project terminology. Args - "<lang1,lang2,...> <audio...> [doc...]" e.g. "ru,en standup.m4a arch.md".
---

# sanar

Local, GPU-accelerated, glossary-primed transcription. One audio timeline, one pass per
language, one verbatim merge.

**The product is verbatim speech.** Everything below serves that. See
[references/merge-protocol.md](references/merge-protocol.md) — read it before Phase 5, it is not optional.

## Arguments

`/sanar <langs> <audio...> [doc...]`

- **langs** — first token, comma-separated ISO codes: `ru,en`. `auto` = let Whisper detect.
  One transcription pass per code, in the order given.
- **audio** — 1..N files. Any container ffmpeg reads (`.mp4 .m4a .mp3 .wav .ogg .opus .flac
  .mkv .mov .webm .aac .wma`). Stitched into one timeline in the order given.
- **doc** — 0..N files (`.md .txt .pdf .rst .adoc .json .yaml .csv .html .py .go .ts ...`).
  Glossary sources. If none are given, mine the current session context instead.

Classify by extension. If a file's role is ambiguous, ask — do not guess.

## Checklist

1. Probe the environment, set up the backend if missing.
2. Stitch audio into one 16 kHz mono WAV.
3. Build the ranked glossary (**before** any transcription).
4. Run one Whisper pass per language.
5. Merge verbatim into the final transcript.

Create one todo per phase. Do not reorder — the glossary must exist before Phase 4.

---

## Phase 1 — Environment

```bash
bash <skill>/scripts/env.sh probe
```

Emits `key=value` lines: OS, WSL flag, arch, chip, RAM, GPU, external GPU, ffmpeg,
chosen `backend`/`device`, venv state. Read it, then:

- `venv=missing` or a missing package → `bash <skill>/scripts/env.sh setup`.
  It builds an isolated venv under `$SANAR_HOME` (default `~/.sanar`) and installs
  `mlx-whisper` (Apple GPU) or `faster-whisper` (CUDA / CPU). It never touches system
  packages.
- `ffmpeg=MISSING` → `setup` prints the exact install command. **Show it to the user and
  wait** — installing system packages is their call, not yours.
- `device=cpu` on a large file → tell the user the expected wall time up front
  (large-v3 on CPU is roughly 1-3x realtime per pass, so `ru,en` on a 1-hour recording is
  hours). Offer `SANAR_MODEL=large-v3-turbo` as the faster, slightly worse alternative and
  let them choose.
- Anything surprising in the probe (WSL without a driver, external GPU, ROCm, tiny RAM) →
  [references/environment.md](references/environment.md) has the specifics and the honest
  limits. Report what you find; do not invent acceleration that isn't there.

First transcription downloads ~3 GB of model weights. Say so before starting it.

## Phase 2 — Stitch

```bash
bash <skill>/scripts/env.sh probe            # once, if not already done
bash <skill>/scripts/stitch.sh sanar-out/audio.wav <audio1> <audio2> ...
```

Writes `sanar-out/audio.wav` plus `sanar-out/audio.chapters.txt` — the offset at which each
source file starts. Keep that map; Phase 5 uses it to attribute segments to source files.

Report the total duration. Sanity-check it against expectations — a 3-second output from a
40-minute input means ffmpeg picked the wrong stream, not a short meeting.

## Phase 3 — Glossary

This phase decides how much of the domain vocabulary survives. Take it seriously.

**Whisper has no dictionary.** It has `initial_prompt`, capped at 223 tokens (~120-150
terms) by the decoder's context window. So the glossary is used twice, differently:

- **top ~200 tokens → the prompt.** Ranked, best-first. The script packs a greedy prefix
  and reports what it dropped.
- **the whole glossary → Phase 5.** No limit there. This is where the long tail gets fixed.

So collect greedily and rank ruthlessly.

### Collect

From documents, if given: read them **in full** (use `pdf-to-markdown` for PDFs). Harvest
every product name, service name, repo, library, framework, protocol, tool, acronym,
`CamelCase`/`snake_case` identifier, env var, metric, table name, job title, team name,
person name, ticket prefix, and domain noun.

If no documents: mine this session's context — files read, commands run, errors seen,
service and branch names, anything the user typed. Say which sources you used.

Then **guess at least 10-15 more that are not written anywhere** but that this audio will
almost certainly contain: the languages and stacks implied by what you saw, the adjacent
tooling (CI, orchestrator, broker, tracer, IaC), the pattern vocabulary (idempotency,
backpressure, saga, circuit breaker), and — for Russian speech — the anglicisms that get
inflected mid-sentence (деплой, мёрдж, реквест, флоу, коммит, рейс, таймаут, ретрай,
пайплайн, фича, релиз, бранч, ревью, шардирование, консистентность). Mark guesses in your
report so the user can veto them.

Aim for 150+ terms total. More is better; the tail is free.

### Rank

Sort by **risk of mangling × cost of being wrong**, best-first:

1. Terms Whisper will certainly destroy: rare acronyms, invented product names, non-word
   identifiers (`ClickHouse`, `pgbouncer`, `KEDA`, `OTel`, `NATS`, `k8s`, `sanar`).
2. Latin terms inside Russian speech — the biggest single win.
3. Homophone traps where the wrong choice reads as plausible prose (`Kafka` / `кафка`,
   `Rust` / `раст`, `Go` / `го`, `Terraform` / `террафор`).
4. Person, team, and service names.
5. Common technical vocabulary Whisper already knows (`Docker`, `Python`, `Kubernetes`) —
   these belong at the bottom; they are the ones to drop.

### Write

```
sanar-out/terms.txt      # one term per line, best-first, no numbering, no comments
sanar-out/glossary.md    # full glossary with the spelling to use + short gloss, grouped
```

`terms.txt` feeds the prompt. `glossary.md` is the authority for Phase 5. Both.

Show the user the top ~30 and the total count before transcribing.

## Phase 4 — Transcribe

One pass per language, sequentially:

```bash
"$SANAR_HOME/venv/bin/python" <skill>/scripts/transcribe.py \
  --audio sanar-out/audio.wav \
  --lang ru \
  --terms sanar-out/terms.txt \
  --out sanar-out/pass-ru.txt
```

Repeat for each language (`pass-en.txt`, ...). Each pass writes `[hh:mm:ss] text` segments
and a sidecar `pass-<lang>.json` with backend, device, model, prompt tokens used/dropped,
detected language, and realtime factor.

Run them one at a time — two large-v3 processes on one GPU is slower than sequential, and
on Apple Silicon it will swap.

Never skip a requested language because an earlier pass "looks fine". Which language wins
per phrase is exactly what Phase 5 decides, and it is not knowable in advance.

If a pass ends early, is much shorter than the others, or contains a phrase repeated many
times, that pass hit a decoder loop. Re-run it with `--no-condition` and keep both outputs.

Report per pass: segment count, prompt terms used vs dropped, wall time, RTF, and anything
anomalous.

## Phase 5 — Merge

**Read [references/merge-protocol.md](references/merge-protocol.md) now.** It is the
contract for this phase.

Inputs: every `pass-*.txt`, every source document, `glossary.md`, `audio.chapters.txt`.

Outputs:

```
sanar-out/transcript.md     # the deliverable: verbatim, timestamped, chaptered
sanar-out/merge-notes.md    # every disagreement between passes and what was chosen, why
```

The one rule that overrides every instinct you have:

> **You are a scribe, not an editor.** Preserve what was said, exactly. If a speaker states
> something factually wrong, keep it wrong. If they contradict the documents, the speaker
> wins. If they contradict themselves, keep both. Fix only what the *microphone and the
> model* got wrong — never what the *speaker* got wrong. Adding, smoothing, summarizing,
> correcting, or "clarifying" is data loss, and it is the single worst failure mode of this
> skill.

Then report: duration, passes, glossary size, where the passes disagreed most, and every
span still marked uncertain.

---

## Notes

- `<skill>` above is this skill's directory. Resolve it once, use absolute paths after.
- All artifacts live in `sanar-out/` in the working directory. Nothing leaves the machine
  except the one-time model download from Hugging Face.
- Env knobs: `SANAR_HOME`, `SANAR_BACKEND` (`mlx`|`faster`), `SANAR_MODEL`,
  `SANAR_DEVICE`, `SANAR_COMPUTE`.
