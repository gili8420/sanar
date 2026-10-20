# Environment, GPUs, and honest limits

`scripts/env.sh probe` reports what it finds. This file explains what the findings mean.

## Backend selection

| Detected | Backend | Device | Speed, large-v3, 1 h audio |
|---|---|---|---|
| macOS + Apple Silicon | `mlx-whisper` | Metal (unified memory) | ~10 min for `ru,en` together (M1 Pro, measured) |
| Linux / WSL2 + NVIDIA | `faster-whisper` | CUDA fp16 | ~2-5 min |
| everything else | `faster-whisper` | CPU int8 | 40 min - 3 h |

Both install as pip wheels into `$SANAR_HOME/venv`. Nothing is compiled.

Override with `SANAR_BACKEND=mlx|faster`, `SANAR_DEVICE`, `SANAR_COMPUTE`.

On MLX one process runs every language: each chunk is encoded once and the encoder output is
decoded per language, `SANAR_BATCH` chunks at a time. faster-whisper uses its
`BatchedInferencePipeline`, which re-encodes per language.

## Diarization

`diarize.py` runs pyannote `community-1` (pyannote.audio 4) from the ungated mirror
`pyannote-community/speaker-diarization-community-1` (CC-BY-4.0): no Hugging Face token,
~30 MB, downloaded once. Device: MPS on Apple Silicon, CUDA when present, else CPU. Measured on
an M1 Pro: ~16x realtime on MPS, ~1x on CPU — the speaker-embedding step is 90% of the time.
Larger `embedding_batch_size` or a coarser step made MPS slower, not faster.

Why this model: it is the open model with the best support for transcripts (exclusive
single-speaker turns, per-speaker embeddings for voice matching, speaker-count hints). DiariZen
scores better in benchmarks but its weights are CC-BY-NC; NVIDIA's Sortformer caps at four
speakers and wants NeMo on a GPU.

Synthetic voices are a poor test of it: on macOS `say` voices it merged two different female
voices into one speaker. Judge it on real recordings.

`objc: Class AVF... is implemented in both` at import time is Homebrew's ffmpeg and PyAV's
bundled one meeting in one process. sanar keeps diarization (torch) and transcription
(mlx / PyAV) in separate processes, so it is noise there.

## `setup` appears to hang with 0% CPU

Almost always the shared uv cache lock. Long-lived `uv tool uvx` processes — MCP servers
started by Claude Code are the usual suspects — hold an exclusive flock on
`~/.cache/uv/.lock` for their entire lifetime, and any `uv pip install` against that cache
blocks on it indefinitely. It looks like a network stall: the process sits at ~0.1% CPU with
no children and never times out.

`env.sh` avoids this by exporting `UV_CACHE_DIR=$SANAR_HOME/uv-cache`, so sanar has its own
cache and its own lock. To confirm the diagnosis on a machine that still stalls:

```bash
lsof ~/.cache/uv/.lock          # every uvx MCP server holding it will be listed
```

## Python version

mlx and ctranslate2 ship wheels for released CPython versions only, usually lagging the
newest by 1-2 releases. `setup` therefore builds its venv on 3.12 when it can find it
(`uv venv --python 3.12`, else `python3.12`, `python3.11`, ...), and only falls back to
`python3` with a warning. A brand-new system Python is the most common setup failure.

## macOS

**Apple Silicon** — the GPU is the right target and MLX uses it. Unified memory means
large-v3 (~3 GB in fp16) is fine on 16 GB, comfortable on 32 GB. No CUDA, no ROCm, and
none is needed.

**External GPU** — the probe checks `system_profiler SPDisplaysDataType` and reports any
non-Apple chipset. Be honest about what it can do:

- On **Apple Silicon**, macOS has no eGPU support at all. A connected Thunderbolt GPU is
  not usable by anything, including this skill.
- On **Intel Macs**, eGPUs work for display and Metal, but there is no CUDA on macOS (Apple
  dropped it after 10.13) and CTranslate2 has no Metal backend. So an AMD eGPU cannot
  accelerate Whisper here. Intel Macs run `faster-whisper` on CPU.

If the probe finds an external GPU, report it and say it will not be used. Do not promise
acceleration that does not exist on this platform.

**AMD / Intel Macs generally** — CPU int8. Warn about wall time before starting.

## WSL2

GPU passthrough needs two things, and only one of them can be done from inside WSL:

1. **Windows-side NVIDIA driver** (≥ 470, current is better). The user installs this on
   Windows. There is no in-WSL substitute, and you must not install a Linux display driver
   inside WSL — that breaks passthrough. If `nvidia-smi` is missing or errors, this is
   almost always the cause.
2. **Userspace CUDA libraries** for CTranslate2 (cuBLAS + cuDNN 9). `setup` installs these
   as pip wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) into the venv, and
   `transcribe.py` preloads them with `RTLD_GLOBAL` so no `LD_LIBRARY_PATH` juggling is
   needed. No CUDA toolkit install, no apt, no sudo.

Checks, in order:

```bash
ls /dev/dxg                 # GPU passthrough device — absent means WSL1 or no driver
nvidia-smi                   # must list the GPU from inside WSL
wsl.exe --version            # (from Windows) WSL2 required; WSL1 has no GPU access
```

Common failures:

- **`/dev/dxg` missing** — WSL1. Convert: `wsl --set-version <distro> 2` from Windows.
- **`nvidia-smi: command not found`** — Windows driver not installed, or the distro was
  started before the driver was. `wsl --shutdown` and reopen.
- **`Could not load libcudnn_ops.so`** — cuDNN wheel missing or a version mismatch. Pin
  `nvidia-cudnn-cu12>=9,<10` for CTranslate2 4.5+; older CTranslate2 needs cuDNN 8.
- **Out of memory on a 6-8 GB card** — use `SANAR_COMPUTE=int8_float16`, or
  `SANAR_MODEL=large-v3-turbo`.
- **Audio path is on `/mnt/c`** — decoding across the 9p filesystem is slow. Copy the file
  into the Linux filesystem first.

## Linux

NVIDIA: driver + the two pip CUDA wheels, same as WSL. `nvidia-smi` must work.

AMD/ROCm: reported by the probe if `rocm-smi` exists, but CTranslate2 has no ROCm backend,
so the run is CPU-bound. Say so plainly rather than implying the card helps.

Intel Arc / oneAPI: not supported by either backend. CPU.

## Resources

- **RAM/VRAM** — large-v3 needs ~3 GB weights plus a few GB of activations. Under 8 GB
  total, expect swapping; suggest `large-v3-turbo`.
- **Disk** — ~3 GB per model in the Hugging Face cache (`~/.cache/huggingface`), fetched
  once on the first pass. Warn the user before that download starts.
- **Batch** — on large-v3 each chunk in a decoding batch holds ~250 MB of cross-attention
  cache (32 layers × 1500 audio frames). The default, 16 with 16 GB of RAM or more and 8
  below, fits next to the 3 GB model; turbo's 4 decoder layers make the batch nearly free.
- **Long audio** — memory use is bounded by the batch, so hours of audio are fine; time is
  the only cost. It scales linearly, so measure one run and extrapolate.

## Model choice

- `large-v3` (default) — best on rare terms and code-switching. This skill exists for
  exactly that, so it is the default.
- `large-v3-turbo` — same 32-layer encoder, 4 decoder layers instead of 32: ~2x faster end
  to end in sanar on Apple Silicon, noticeably weaker on rare proper nouns, and on mixed ru/en
  speech its `en` pass drifts into Russian and drops English. A time concession, not a free
  win. Offer it; do not silently substitute it.
- Anything smaller defeats the purpose of a glossary-primed run.

Quantized MLX weights do not help here: with batched decoding the GPU is compute-bound, and
`whisper-large-v3-mlx-8bit` / `-4bit` ran slower than fp16 (67 s / 70 s vs 56 s on the same
6-minute test, same recall).

Set via `SANAR_MODEL`. On MLX the short names map to `mlx-community` repos
(`large-v3` → `whisper-large-v3-mlx`, `large-v3-turbo` / `turbo` → `whisper-large-v3-turbo`);
a name with a `/` is used as a repo as is. On faster-whisper use the plain `large-v3` /
`large-v3-turbo` names or a local CTranslate2 dir.

## Chunks, prompts and decoder loops

Every chunk (at most 28 s, cut along speaker turns) is decoded on its own with the glossary
as its prompt. There is no conditioning on the previous chunk's text, so the glossary never
scrolls out of the window and a degenerate loop cannot spread past one chunk.

A chunk whose output looks like a loop (compression ratio > 2.4), has low confidence
(avg logprob < -1) or hits the token limit is redone automatically with temperature fallback
and listed under `fallback` in the pass sidecar. Silence and a recited glossary (Whisper
reading the prompt back over near-silence) are dropped and listed under `dropped`.
