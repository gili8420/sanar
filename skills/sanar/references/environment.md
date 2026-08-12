# Environment, GPUs, and honest limits

`scripts/env.sh probe` reports what it finds. This file explains what the findings mean.

## Backend selection

| Detected | Backend | Device | Speed, large-v3, 1 h audio |
|---|---|---|---|
| macOS + Apple Silicon | `mlx-whisper` | Metal (unified memory) | ~4-8 min |
| Linux / WSL2 + NVIDIA | `faster-whisper` | CUDA fp16 | ~2-5 min |
| everything else | `faster-whisper` | CPU int8 | 40 min - 3 h |

Both install as pip wheels into `$SANAR_HOME/venv`. Nothing is compiled.

Override with `SANAR_BACKEND=mlx|faster`, `SANAR_DEVICE`, `SANAR_COMPUTE`.

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
- **Long audio** — memory use is bounded by the 30 s window, so hours of audio are fine;
  time is the only cost. It scales linearly, so measure one pass and extrapolate.

## Model choice

- `large-v3` (default) — best on rare terms and code-switching. This skill exists for
  exactly that, so it is the default.
- `large-v3-turbo` — ~4-8x faster, noticeably weaker on rare proper nouns. A time
  concession, not a free win. Offer it; do not silently substitute it.
- Anything smaller defeats the purpose of a glossary-primed run.

Set via `SANAR_MODEL`. On MLX use the `mlx-community/whisper-*` repo names; on
faster-whisper use the plain `large-v3` / `large-v3-turbo` names or a local CTranslate2 dir.

## Decoder loops

Both backends condition each 30 s window on the previous window's text — that is how a
correctly-spelled glossary term propagates through the whole file, and it is why this is on
by default. The cost is a rare degenerate loop where one phrase repeats for minutes.

`faster-whisper` mitigates it with VAD plus temperature fallback and a prompt reset.
`mlx-whisper` has temperature fallback only.

If a pass loops: re-run it with `--no-condition` and keep **both** outputs. The looping pass
is often still correct up to the loop, and the merge can use that part.
