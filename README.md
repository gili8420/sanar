<p align="center">
  <img src="assets/logo.png" alt="sanar" width="420">
</p>

<p align="center">
  Context-aware <strong>local</strong> transcription for Claude Code.<br>
  Nothing leaves your machine except a one-time model download.
</p>

```
/sanar ru,en standup1.m4a standup2.m4a arch.md glossary.md
```

1. **Probe** the machine — macOS / Linux / WSL2, and whether there's a usable GPU.
2. **Stitch** all audio into one 16 kHz mono timeline, keeping a map of source offsets.
3. **Build a glossary** from your documents (or from the session context) and rank it.
4. **Transcribe** with Whisper large-v3 — one pass per language, glossary in the prompt.
5. **Merge verbatim** into a single transcript, plus an audit trail of every decision.

## Why multiple passes

For speech that is mostly Russian with dense English terminology, the `ru` pass gets the
connective speech right and mangles the Latin terms; the `en` pass does the opposite. Which
one wins for a given phrase is not knowable in advance, so sanar runs both and merges at
word granularity.

## Why a glossary

Whisper has no dictionary. It has `initial_prompt`, hard-capped at 223 tokens (~120-150
terms) by the decoder's context window. sanar uses the glossary twice:

- the **ranked top ~200 tokens** go into the prompt, where they steer decoding and then
  self-propagate — each 30 s window is conditioned on the previous window's text, so a term
  spelled correctly once tends to stay correct;
- the **full glossary** goes into the merge phase, where there is no token limit and the long
  tail gets fixed.

## Install

**As a plugin** (recommended — gives you `/sanar` and keeps updates one command away):

```
/plugin marketplace add <your-gh-user>/sanar
/plugin install sanar@sanar
```

**As a plain skill:**

```bash
git clone --depth 1 https://github.com/<your-gh-user>/sanar ~/.sanar-src
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
  (plus cuBLAS/cuDNN wheels) on CUDA and CPU. Nothing is compiled.
- ~3 GB of disk for the large-v3 weights, fetched once from Hugging Face.

Verify the whole pipeline without downloading large-v3 (uses `whisper-tiny`, ~75 MB):

```bash
bash ~/.claude/skills/sanar/scripts/selftest.sh
```

## Usage

```
/sanar <langs> <audio...> [docs...]
```

| Part | Meaning |
|---|---|
| `<langs>` | comma-separated, one pass each: `ru,en`, `ru`, `en,de,auto` |
| `<audio...>` | 1..N files, any container ffmpeg reads; stitched in the order given |
| `[docs...]` | 0..N term sources. Omit to mine the current session context instead |

Examples:

```
/sanar ru                          interview.mp3
/sanar ru,en                       call1.m4a call2.m4a design-doc.md adr-014.md
/sanar en,auto                     recording.mkv
/sanar ru,en                       planning.mp4        # no docs: uses session context
```

Output lands in `sanar-out/` next to where you ran it:

```
sanar-out/
  audio.wav              stitched 16 kHz mono
  audio.chapters.txt     which source file starts at which offset
  terms.txt              ranked glossary fed to the prompt
  glossary.md            full glossary with canonical spellings
  pass-ru.txt            one per language: [hh:mm:ss] segments
  pass-ru.txt.json       backend, device, RTF, prompt terms used/dropped
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

Whisper large-v3, roughly, per pass on 1 hour of audio:

| Machine | Backend | Wall |
|---|---|---|
| Apple Silicon (M1 Pro / M2 / M3) | mlx-whisper, Metal | ~4-8 min |
| NVIDIA (Linux or WSL2), fp16 | faster-whisper, CUDA | ~2-5 min |
| CPU only | faster-whisper, int8 | 40 min - 3 h |

Multiply by the number of languages. On CPU, sanar warns you before starting and offers
`SANAR_MODEL=large-v3-turbo` — 4-8x faster, weaker on rare proper nouns, which is exactly
what this tool is trying to get right. Your call, not a silent substitution.

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

The scripts also run standalone, without Claude:

```bash
bash skills/sanar/scripts/env.sh probe
bash skills/sanar/scripts/stitch.sh sanar-out/audio.wav a.m4a b.m4a
~/.sanar/venv/bin/python skills/sanar/scripts/transcribe.py \
  --audio sanar-out/audio.wav --lang ru --terms sanar-out/terms.txt \
  --out sanar-out/pass-ru.txt
```

## Layout

```
.claude-plugin/    plugin + marketplace manifests
commands/sanar.md  slash-command shim
skills/sanar/
  SKILL.md         the five-phase workflow Claude follows
  references/      merge protocol, environment notes
  scripts/         env.sh, stitch.sh, transcribe.py, selftest.sh
```

## Limits

- Prompt priming is capped at 223 tokens by Whisper itself. Terms beyond the cap are handled
  in the merge, not in decoding. There is no way around this short of fine-tuning.
- No diarization. Speaker labels appear only when the passes genuinely support them; sanar
  will not invent them.
- Merge quality scales with Claude's context. Multi-hour recordings are better done in
  chunks than in one heroic merge.
- Sequential passes by design — two large-v3 processes on one GPU is slower than one at a
  time, and on Apple Silicon it swaps.

## License

MIT
