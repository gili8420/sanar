#!/usr/bin/env bash
# sanar environment: `probe` reports hardware and backend choice, `setup` builds the venv.
# Never installs system packages — it prints the command and lets the human decide.
set -uo pipefail

SANAR_HOME="${SANAR_HOME:-$HOME/.sanar}"
VENV="$SANAR_HOME/venv"
PY="$VENV/bin/python"

# Long-lived `uv tool uvx` processes (MCP servers, commonly running in Claude Code) hold an
# exclusive flock on the shared uv cache for their entire lifetime, so `uv pip install`
# against the default cache blocks forever. Own cache dir, own lock.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SANAR_HOME/uv-cache}"

have() { command -v "$1" >/dev/null 2>&1; }
kv() { printf '%s=%s\n' "$1" "$2"; }

# ---------- detection ----------

is_wsl() { [[ -r /proc/version ]] && grep -qiE 'microsoft|wsl' /proc/version; }

nvidia_name() {
  have nvidia-smi || return 1
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
}

# macOS: every GPU chipset macOS knows about. Anything not "Apple" is discrete/external.
mac_gpus() { system_profiler SPDisplaysDataType 2>/dev/null | sed -n 's/^ *Chipset Model: *//p'; }

ram_gb() {
  case "$(uname -s)" in
    Darwin) awk -v b="$(sysctl -n hw.memsize 2>/dev/null || echo 0)" 'BEGIN{printf "%d", b/1073741824}' ;;
    *) awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0 ;;
  esac
}

cores() { getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0; }

chip() {
  case "$(uname -s)" in
    Darwin) sysctl -n machdep.cpu.brand_string 2>/dev/null ;;
    *) sed -n 's/^model name[ \t]*: *//p' /proc/cpuinfo 2>/dev/null | head -1 ;;
  esac
}

# Resolve backend/device without printing, so both probe and setup share one decision.
decide() {
  BACKEND="${SANAR_BACKEND:-}"
  local os arch
  os=$(uname -s); arch=$(uname -m)
  if [[ -z "$BACKEND" ]]; then
    if [[ "$os" == Darwin && "$arch" == arm64 ]]; then BACKEND=mlx; else BACKEND=faster; fi
  fi
  if [[ "$BACKEND" == mlx ]]; then
    DEVICE=metal
  elif nvidia_name >/dev/null 2>&1; then
    DEVICE=cuda
  else
    DEVICE=cpu
  fi
  DEVICE="${SANAR_DEVICE:-$DEVICE}"
}

probe() {
  local os arch ext="none" gpus g
  os=$(uname -s); arch=$(uname -m)
  decide

  kv os "$os"
  kv kernel "$(uname -r)"
  kv arch "$arch"
  kv wsl "$(is_wsl && echo 1 || echo 0)"
  kv chip "$(chip)"
  kv cores "$(cores)"
  kv ram_gb "$(ram_gb)"

  if [[ "$os" == Darwin ]]; then
    kv macos "$(sw_vers -productVersion 2>/dev/null)"
    gpus=$(mac_gpus | paste -sd'; ' -)
    kv gpu "${gpus:-unknown}"
    while read -r g; do
      [[ -n "$g" && "$g" != Apple* ]] && ext="$g"
    done < <(mac_gpus)
    # Apple Silicon macOS has no eGPU support at all; Intel macOS has no CUDA and
    # CTranslate2 has no Metal. Either way an external GPU cannot serve Whisper here.
    kv external_gpu "$ext"
    [[ "$ext" != none ]] && kv external_gpu_usable "no (see references/environment.md)" || true
  else
    kv gpu "$(nvidia_name || { have lspci && lspci 2>/dev/null | sed -n 's/.*VGA compatible controller: //p' | head -1; } || echo unknown)"
    kv nvidia_smi "$(nvidia_name || echo absent)"
    kv rocm "$(have rocm-smi && rocm-smi --showproductname 2>/dev/null | sed -n 's/^GPU\[0\].*: *//p' | head -1 || echo absent)"
    is_wsl && kv wsl_dxg "$([[ -e /dev/dxg ]] && echo present || echo MISSING)"
  fi

  kv ffmpeg "$(command -v ffmpeg || echo MISSING)"
  kv ffprobe "$(command -v ffprobe || echo MISSING)"
  kv backend "$BACKEND"
  kv device "$DEVICE"
  kv model "${SANAR_MODEL:-large-v3}"
  kv sanar_home "$SANAR_HOME"

  if [[ -x "$PY" ]]; then
    kv venv ready
    kv venv_python "$("$PY" -V 2>&1)"
    local pkg=faster_whisper; [[ "$BACKEND" == mlx ]] && pkg=mlx_whisper
    if "$PY" -c "import $pkg" 2>/dev/null; then kv backend_pkg ready; else kv backend_pkg MISSING; fi
    if "$PY" -c "import pyannote.audio" 2>/dev/null; then kv diarize_pkg ready; else kv diarize_pkg MISSING; fi
  else
    kv venv missing
  fi
}

# ---------- setup ----------

# mlx / ctranslate2 wheels lag the newest CPython by a release or two.
pick_python() {
  local v
  if have uv; then
    for v in 3.12 3.11 3.13; do
      uv venv --quiet --python "$v" "$VENV" 2>/dev/null && { echo "uv:$v"; return 0; }
    done
    uv venv --quiet "$VENV" 2>/dev/null && { echo "uv:default"; return 0; }
  fi
  for v in python3.12 python3.11 python3.13 python3.10 python3; do
    have "$v" || continue
    "$v" -m venv "$VENV" 2>/dev/null && { echo "$v"; return 0; }
  done
  return 1
}

setup() {
  decide
  mkdir -p "$SANAR_HOME"

  if ! have ffmpeg || ! have ffprobe; then
    echo "ffmpeg is required and missing. Install it, then re-run setup:" >&2
    case "$(uname -s)" in
      Darwin) echo "  brew install ffmpeg" >&2 ;;
      *) if have apt-get; then echo "  sudo apt-get update && sudo apt-get install -y ffmpeg" >&2
         elif have dnf; then echo "  sudo dnf install -y ffmpeg" >&2
         elif have pacman; then echo "  sudo pacman -S --needed ffmpeg" >&2
         else echo "  install ffmpeg via your package manager" >&2; fi ;;
    esac
    exit 3
  fi

  if [[ ! -x "$PY" ]]; then
    echo "creating venv at $VENV" >&2
    local src; src=$(pick_python) || { echo "no usable python3 found (need 3.10-3.13)" >&2; exit 4; }
    echo "  interpreter: $src" >&2
  fi

  # pyannote.audio: diarization (pulls torch; runs on MPS / CUDA / CPU).
  local pkgs=("pyannote.audio>=4,<5")
  # Pinned: transcribe.py calls mlx-whisper's decode/find_alignment internals and
  # faster-whisper's batched clip_timestamps.
  if [[ "$BACKEND" == mlx ]]; then
    pkgs+=("mlx-whisper>=0.4.3,<0.5")
  else
    pkgs+=("faster-whisper>=1.1,<2")
    # WSL/Linux CUDA: userspace cuBLAS+cuDNN as wheels. transcribe.py preloads them,
    # so no LD_LIBRARY_PATH and no CUDA toolkit install.
    [[ "$DEVICE" == cuda ]] && pkgs+=("nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9,<10")
  fi

  echo "installing: ${pkgs[*]}" >&2
  if have uv; then
    uv pip install --quiet --python "$PY" --upgrade "${pkgs[@]}" || exit 5
  else
    "$PY" -m pip install --quiet --upgrade pip || exit 5
    "$PY" -m pip install --quiet --upgrade "${pkgs[@]}" || exit 5
  fi

  echo "ready: backend=$BACKEND device=$DEVICE python=$PY" >&2
  probe
}

case "${1:-probe}" in
  probe) probe ;;
  setup) setup ;;
  *) echo "usage: env.sh [probe|setup]" >&2; exit 2 ;;
esac
