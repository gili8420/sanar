<p align="center">
  <img src="assets/logo.png" alt="sanar" width="420">
</p>

<p align="center">
  Context-aware <strong>local</strong> transcription for Claude Code.<br>
  Nothing leaves your machine except a one-time model download.
</p>

```
/sanar ru,en standup1.m4a standup2.m4a arch.md glossary.md speakers=4
```

1. **Probe** the machine — macOS / Linux / WSL2, and whether there's a usable GPU.
2. **Stitch** all audio into one 16 kHz mono timeline, keeping a map of source offsets.
3. **Diarize** — who spoke when, once for the whole timeline, in the background.
4. **Build a glossary** from your documents (or from the session context) and rank it.
5. **Transcribe** with Whisper large-v3 — one pass per language, glossary in every chunk's
   prompt, every word labelled with its speaker.
6. **Merge verbatim** into a single transcript, plus an audit trail of every decision.

## Why multiple passes

For speech that is mostly Russian with dense English terminology, the `ru` pass gets the
connective speech right and mangles the Latin terms; the `en` pass does the opposite. Which
one wins for a given phrase is not knowable in advance, so sanar runs both and merges at
word granularity.

## Why a glossary

Whisper has no dictionary. It has `initial_prompt`, hard-capped at 223 tokens (~120-150
terms) by the decoder's context window. sanar uses the glossary twice:

- the **ranked top ~200 tokens** go into the prompt of every chunk (≤28 s, cut along speaker
  turns), so the glossary steers the whole recording, not just its first minute;
- the **full glossary** goes into the merge phase, where there is no token limit and the long
  tail gets fixed.

## Who spoke

pyannote `community-1` labels speaker turns once for the whole timeline (`S1`, `S2`, ... in
order of appearance), and each transcribed word gets the label of the turn it falls in. Give
the speaker count when you know it (`speakers=4` or `speakers=3-5`) — counting is where
diarization fails most. Give `voices=DIR` with a 10-30 s clip per person (`Anna.m4a`,
`Миша.wav`) and matching speakers get names, with a similarity score and a `matched` /
`review` band. A name Claude infers from the conversation ("Миша, что скажешь?") goes into
the speaker table with its evidence, never onto transcript lines.

## Install

**As a plugin** (recommended — gives you `/sanar` and keeps updates one command away):

```
/plugin marketplace add gili8420/sanar
/plugin install sanar@sanar
```

A plugin skill is invoked as `/sanar:sanar ...`; a plain skill (below) as `/sanar ...`.

**As a plain skill:**

```bash
git clone --depth 1 https://github.com/gili8420/sanar ~/.sanar-src
mkdir -p ~/.claude/skills
ln -sfn ~/.sanar-src/skills/sanar ~/.claude/skills/sanar
```

Update with `git -C ~/.sanar-src pull`.

**Per project** instead of globally: symlink into `.claude/skills/sanar` in the repo.

### Prerequisites

- **ffmpeg** — `brew install ffmpeg`, `sudo apt install ffmpeg`. The only system dependency;
  sanar prints the command rather than installing it for you.
- **Python 3.10-3.13** — sanar builds its own venv in `~/.sanar`, preferring 3.12 (mlx and
  ctranslate2 wheels lag the newest CPython). `uv` is used if present.
- Everything else installs on first run: `mlx-whisper` on Apple Silicon, `faster-whisper`
  (plus cuBLAS/cuDNN wheels) on CUDA and CPU, and `pyannote.audio` (with PyTorch) for
  diarization everywhere. Nothing is compiled.
- ~3 GB of disk for the large-v3 weights and ~30 MB for the diarization models, fetched once
  from Hugging Face. No Hugging Face token: sanar uses the ungated `pyannote-community` mirror.

Verify the whole pipeline without downloading large-v3 (uses `whisper-tiny`, ~75 MB):

```bash
bash ~/.claude/skills/sanar/scripts/selftest.sh
```

## Usage

```
/sanar <langs> <audio...> [docs...] [speakers=N|A-B] [voices=DIR]
```

| Part | Meaning |
|---|---|
| `<langs>` | comma-separated, one pass each: `ru,en`, `ru`, `en,de,auto` |
| `<audio...>` | 1..N files, any container ffmpeg reads; stitched in the order given |
| `[docs...]` | 0..N term sources. Omit to mine the current session context instead |
| `speakers=` | known speaker count `4`, or bounds `3-5` |
| `voices=` | folder of `<Name>.<audio>` clips, 10-30 s each, to name the speakers |

Examples:

```
/sanar ru                          interview.mp3 speakers=2
/sanar ru,en                       call1.m4a call2.m4a design-doc.md adr-014.md
/sanar en,auto                     recording.mkv voices=~/voices
/sanar ru,en                       planning.mp4        # no docs: uses session context
```

Output lands in `sanar-out/` next to where you ran it:

```
sanar-out/
  audio.wav              stitched 16 kHz mono
  audio.chapters.txt     which source file starts at which offset
  speakers.json          speaker turns, overlaps, seconds per speaker, voice matches
  terms.txt              ranked glossary fed to the prompt
  glossary.md            full glossary with canonical spellings
  pass-ru.txt            one per language: @cNNN chunks of [hh:mm:ss] S1: lines
  pass-ru.txt.json       backend, device, timings, prompt terms, redone/dropped chunks
  transcript.md          the deliverable
  merge-notes.md         every disagreement, choice, and reason
```

## Verbatim is the point

The merge phase runs under one hard rule: **preserve what was said, exactly.**

Claude may fix what the model misheard. Claude may not fix what the speaker got wrong. If a
speaker states something factually false, contradicts your documents, repeats themselves, or
leaves a sentence unfinished, that survives into the transcript untouched — a transcript's
whole value is that it is not the transcriber's opinion. Spans that stay ambiguous are marked
`[?a|b]` or `[unintelligible ~3s]` rather than silently guessed, and `merge-notes.md` records
every decision so it can be checked without re-listening.

Full rules: [`skills/sanar/references/merge-protocol.md`](skills/sanar/references/merge-protocol.md).

## Performance

Measured on an M1 Pro (16 GB), on a scripted 6.4-minute ru/en meeting with three voices,
`ru,en`, model load excluded. Recall is the share of the script's words that reach at least
one pass — what the merge has to work with:

| Run | Wall | ≈ per hour | Russian recall | English recall |
|---|---|---|---|---|
| sanar 0.1 (sequential passes) | 143 s | 22 min | 91% | 98% |
| **sanar 0.2, large-v3** | **65 s** | **10 min** | 93% | 98% |
| sanar 0.2, `large-v3-turbo` | 36 s | 6 min | 92% | 76% |
| diarization, MPS (runs while Claude builds the glossary) | 24 s | 4 min | — | — |

Where the time went: one encoder pass per chunk shared by every language, and batched
decoding (batch 4 → 18 took large-v3 from 101 s to 56 s). Quantized 8/4-bit weights were
slower, not faster.

Other machines, not re-measured for 0.2:

| Machine | Backend | Wall per language, 1 h audio |
|---|---|---|
| NVIDIA (Linux or WSL2), fp16 | faster-whisper, CUDA, batched | ~2-5 min |
| CPU only | faster-whisper, int8, batched | 40 min - 3 h |

On CPU, sanar warns you before starting and offers `SANAR_MODEL=large-v3-turbo` — faster,
weaker on rare proper nouns, which is exactly what this tool is trying to get right. Your
call, not a silent substitution.

**GPU honesty:** on Apple Silicon macOS has no eGPU support at all, and on Intel Macs there
is no CUDA and no Metal backend in CTranslate2. sanar reports an attached external GPU and
tells you it will not be used, rather than implying acceleration that doesn't exist. WSL2
needs the **Windows-side** NVIDIA driver — that part can't be installed from inside WSL;
everything else sanar handles as pip wheels. Details and troubleshooting:
[`skills/sanar/references/environment.md`](skills/sanar/references/environment.md).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SANAR_HOME` | `~/.sanar` | venv location |
| `SANAR_BACKEND` | auto | `mlx` \| `faster` |
| `SANAR_MODEL` | `large-v3` | e.g. `large-v3-turbo`, or a local CTranslate2 dir |
| `SANAR_DEVICE` | auto | faster-whisper: `cuda` \| `cpu` |
| `SANAR_COMPUTE` | auto | faster-whisper: `float16`, `int8`, `int8_float16` |
| `SANAR_BATCH` | `16` (`8` under 16 GB RAM) | chunks decoded together; ~250 MB each on large-v3 |

The scripts also run standalone, without Claude:

```bash
bash skills/sanar/scripts/env.sh probe
bash skills/sanar/scripts/stitch.sh sanar-out/audio.wav a.m4a b.m4a
~/.sanar/venv/bin/python skills/sanar/scripts/diarize.py \
  --audio sanar-out/audio.wav --out sanar-out/speakers.json --speakers 2-4
~/.sanar/venv/bin/python skills/sanar/scripts/transcribe.py \
  --audio sanar-out/audio.wav --speakers sanar-out/speakers.json --lang ru,en \
  --terms sanar-out/terms.txt --out-dir sanar-out
```

## Layout

```
.claude-plugin/    plugin + marketplace manifests
skills/sanar/
  SKILL.md         the six-phase workflow Claude follows
  references/      merge protocol, environment notes
  scripts/         env.sh, stitch.sh, diarize.py, transcribe.py, selftest.sh
```

## Limits

- Prompt priming is capped at 223 tokens by Whisper itself. Terms beyond the cap are handled
  in the merge, not in decoding. There is no way around this short of fine-tuning.
- Diarization is pyannote's, with its limits: speaker counting is the weak spot (give
  `speakers=`), overlapping speech goes to one speaker, and similar voices can merge. Claude
  may move a speaker boundary only for 1-5 words at a turn edge, and never invents a speaker.
- Merge quality scales with Claude's context. Multi-hour recordings are better done in
  chunks than in one heroic merge.

## License

MIT
