---
name: sanar
description: Context-aware local transcription with speaker labels - Whisper large-v3 primed with a glossary, pyannote diarization, one pass per language, verbatim merge. Use when the user runs /sanar, or asks to transcribe audio/video locally, label who spoke, run multi-language Whisper passes, or prime a transcription with project terminology. Args - "<lang1,lang2,...> <audio...> [doc...] [speakers=N|A-B] [voices=DIR]" e.g. "ru,en standup.m4a arch.md speakers=4".
argument-hint: <lang1,lang2> <audio...> [doc...] [speakers=N|A-B] [voices=DIR]
---

# sanar

Local, GPU-accelerated, glossary-primed transcription with speaker labels. One audio
timeline, one diarization, one pass per language, one verbatim merge.

**The product is verbatim speech.** Everything below serves that. See
[references/merge-protocol.md](references/merge-protocol.md) — read it before Phase 6, it is not optional.

## Arguments

`/sanar <langs> <audio...> [doc...] [speakers=N|A-B] [voices=DIR]`

- **langs** — first token, comma-separated ISO codes: `ru,en`. `auto` = let Whisper detect.
  One transcription pass per code.
- **audio** — 1..N files. Any container ffmpeg reads (`.mp4 .m4a .mp3 .wav .ogg .opus .flac
  .mkv .mov .webm .aac .wma`). Stitched into one timeline in the order given.
- **doc** — 0..N files (`.md .txt .pdf .rst .adoc .json .yaml .csv .html .py .go .ts ...`).
  Glossary sources. If none are given, mine the current session context instead.
- **speakers=3** or **speakers=2-5** — the known number of speakers, or bounds. Pass it
  whenever the user knows it: counting speakers is the weakest part of diarization.
- **voices=DIR** — a folder of `<Name>.<audio>` clips, 10-30 s of one person each. Speakers
  whose voice matches get that name.

Classify by extension. If a file's role is ambiguous, ask — do not guess.

## Checklist

1. Probe the environment, set up the backend if missing.
2. Stitch audio into one 16 kHz mono WAV.
3. Start diarization in the background.
4. Build the ranked glossary while it runs (**before** any transcription).
5. Transcribe: one command for every language.
6. Merge verbatim into the final transcript.

Create one todo per phase. Phase 5 needs both 3 and 4 finished.

---

## Phase 1 — Environment

```bash
bash <skill>/scripts/env.sh probe
```

Emits `key=value` lines: OS, WSL flag, arch, chip, RAM, GPU, external GPU, ffmpeg,
chosen `backend`/`device`, `sanar_home`, venv and package state. Read it, then:

- `venv=missing`, `backend_pkg=MISSING` or `diarize_pkg=MISSING` →
  `bash <skill>/scripts/env.sh setup`. It builds an isolated venv under `$SANAR_HOME`
  (default `~/.sanar`) and installs `mlx-whisper` (Apple GPU) or `faster-whisper` (CUDA /
  CPU), plus `pyannote.audio` for diarization (it pulls PyTorch). It never touches system
  packages.
- `ffmpeg=MISSING` → `setup` prints the exact install command. **Show it to the user and
  wait** — installing system packages is their call, not yours.
- `device=cpu` on a large file → tell the user the expected wall time up front: large-v3 on
  CPU is roughly 1-3x realtime per language, and diarization on CPU is about 1x realtime, so
  `ru,en` on a 1-hour recording is hours. Offer `SANAR_MODEL=large-v3-turbo` as the faster,
  slightly worse alternative and let them choose.
- Anything surprising in the probe (WSL without a driver, external GPU, ROCm, tiny RAM) →
  [references/environment.md](references/environment.md) has the specifics and the honest
  limits. Report what you find; do not invent acceleration that isn't there.

The first run downloads ~3 GB of Whisper weights and ~30 MB of diarization models. Say so
before starting it.

Below, `$PY` is `<sanar_home>/venv/bin/python` with `sanar_home` taken from the probe output
(`$SANAR_HOME` is usually not set in your shell).

## Phase 2 — Stitch

```bash
bash <skill>/scripts/stitch.sh sanar-out/audio.wav <audio1> <audio2> ...
```

Writes `sanar-out/audio.wav` plus `sanar-out/audio.chapters.txt` — the offset at which each
source file starts. Keep that map; Phase 6 uses it to attribute lines to source files.

Report the total duration. Sanity-check it against expectations — a 3-second output from a
40-minute input means ffmpeg picked the wrong stream, not a short meeting.

## Phase 3 — Diarize (background)

```bash
"$PY" <skill>/scripts/diarize.py --audio sanar-out/audio.wav --out sanar-out/speakers.json \
  [--speakers 3 | --speakers 2-5] [--voices <dir>]
```

**Start it in the background and go straight to Phase 4.** It needs no glossary, and the
glossary takes you longer than diarization takes the GPU (about 16x realtime on an M1 Pro,
i.e. ~4 min per hour of audio).

`speakers.json` holds the exclusive speaker turns (one speaker at a time, labelled `S1`,
`S2`, ... in order of first appearance), overlap spans, seconds per speaker and, with
`--voices`, each speaker's closest enrolled voice: `matched`, `review` or `unmatched`, with the
cosine similarity.

When it finishes, report: speakers found, seconds each, voice matches with band and
similarity. Two sanity checks:

- One speaker in a recording the user called a meeting → the count is wrong; rerun with
  `--speakers`.
- A speaker with only a few seconds in total is usually a split-off of someone else or a lone
  backchannel. Mention it; do not merge speakers yourself.

## Phase 4 — Glossary

This phase decides how much of the domain vocabulary survives. Take it seriously.

**Whisper has no dictionary.** It has a prompt, capped at 223 tokens (~120-150 terms) by the
decoder's context window. So the glossary is used twice, differently:

- **top ~200 tokens → the prompt of every chunk.** Ranked, best-first. The script packs a
  greedy prefix and reports what it dropped.
- **the whole glossary → Phase 6.** No limit there. This is where the long tail gets fixed.

So collect greedily and rank ruthlessly.

### Collect

From documents, if given: read them **in full** (use `pdf-to-markdown` for PDFs). Harvest
every product name, service name, repo, library, framework, protocol, tool, acronym,
`CamelCase`/`snake_case` identifier, env var, metric, table name, job title, team name,
person name, ticket prefix, and domain noun. Names from `voices=` belong here too.

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

`terms.txt` feeds the prompt. `glossary.md` is the authority for Phase 6. Both.

Show the user the top ~30 and the total count before transcribing.

## Phase 5 — Transcribe

Wait for Phase 3, then one command for all languages:

```bash
"$PY" <skill>/scripts/transcribe.py \
  --audio sanar-out/audio.wav --speakers sanar-out/speakers.json \
  --lang ru,en --terms sanar-out/terms.txt --out-dir sanar-out
```

The script cuts the audio into chunks of at most 28 s along the speaker turns, encodes each
chunk once, and decodes it once per language in batches, with the glossary as the prompt of
every chunk. Per language it writes:

- `pass-<lang>.txt` — `@cNNN hh:mm:ss-hh:mm:ss` chunk headers (the same chunks in every pass),
  then `[hh:mm:ss] S1: text` lines, each word labelled with the speaker whose turn holds it;
- `pass-<lang>.txt.json` — model, device, chunks, timings, realtime factor, prompt terms used
  and dropped, detected languages, and the `fallback` and `dropped` chunk lists.

Chunks that fail Whisper's quality checks (repetition, low confidence, hitting the token
limit) are redone automatically with temperature fallback and listed under `fallback`.
Chunks judged silence or a recited glossary are left empty and listed under `dropped`, with
the text that was dropped. Report both.

Never skip a requested language because an earlier pass "looks fine". Which language wins
per phrase is exactly what Phase 6 decides, and it is not knowable in advance. A pass that is
nearly empty in some chunks is normal: those chunks are the other language's speech.

Report: chunks, wall time, realtime factor, prompt terms used vs dropped, fallback and dropped
counts, and anything anomalous.

Speed knobs, if the user asks: `SANAR_BATCH` (chunks decoded together; default 16 with
16 GB of RAM or more, else 8; ~250 MB each on large-v3), and `SANAR_MODEL=large-v3-turbo` —
about 2x faster on Apple Silicon, but weaker on rare proper nouns and, on mixed ru/en
speech, its `en` pass slides into Russian and loses English (English recall 76-80% vs 97-98%
on a scripted meeting). Offer it; never substitute it silently.

## Phase 6 — Merge

**Read [references/merge-protocol.md](references/merge-protocol.md) now.** It is the
contract for this phase.

Inputs: every `pass-*.txt` and its sidecar, `speakers.json`, every source document,
`glossary.md`, `audio.chapters.txt`.

Outputs:

```
sanar-out/transcript.md     # the deliverable: verbatim, timestamped, chaptered, speaker-labelled
sanar-out/merge-notes.md    # every disagreement between passes and what was chosen, why
```

The one rule that overrides every instinct you have:

> **You are a scribe, not an editor.** Preserve what was said, exactly. If a speaker states
> something factually wrong, keep it wrong. If they contradict the documents, the speaker
> wins. If they contradict themselves, keep both. Fix only what the *microphone and the
> model* got wrong — never what the *speaker* got wrong. Adding, smoothing, summarizing,
> correcting, or "clarifying" is data loss, and it is the single worst failure mode of this
> skill.

Then report: duration, passes, speakers (with names and how they were established), glossary
size, where the passes disagreed most, every moved speaker boundary, and every span still
marked uncertain.

---

## Notes

- `<skill>` above is this skill's directory. Resolve it once, use absolute paths after.
- All artifacts live in `sanar-out/` in the working directory. Nothing leaves the machine
  except the one-time model downloads from Hugging Face.
- Env knobs: `SANAR_HOME`, `SANAR_BACKEND` (`mlx`|`faster`), `SANAR_MODEL`,
  `SANAR_DEVICE`, `SANAR_COMPUTE`, `SANAR_BATCH`.
