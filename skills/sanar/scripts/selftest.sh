#!/usr/bin/env bash
# End-to-end smoke test with whisper-tiny (~75 MB) instead of large-v3 (~3 GB).
# Proves: probe, setup, stitch offsets, glossary packing, backend call, output format.
# usage: bash selftest.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
export SANAR_HOME="${SANAR_HOME:-$HOME/.sanar}"
export SANAR_SCRIPTS="$here"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== probe"
bash "$here/env.sh" probe | tee "$work/probe.txt"
backend=$(sed -n 's/^backend=//p' "$work/probe.txt")
grep -q '^ffmpeg=/' "$work/probe.txt" || fail "ffmpeg missing"

echo "== setup"
bash "$here/env.sh" setup >/dev/null || fail "setup"
PY="$SANAR_HOME/venv/bin/python"
[[ -x "$PY" ]] || fail "no venv python"

echo "== fixtures"
# Two clips so the chapter map has something to prove. `say` on macOS, a tone elsewhere.
if command -v say >/dev/null; then
  say -o "$work/a.aiff" "Kubernetes and ClickHouse in the same cluster."
  say -o "$work/b.aiff" "We merged the pull request and deployed it."
else
  ffmpeg -nostdin -y -loglevel error -f lavfi -i "sine=f=440:d=3" "$work/a.wav"
  ffmpeg -nostdin -y -loglevel error -f lavfi -i "sine=f=880:d=2" "$work/b.wav"
fi
a=$(ls "$work"/a.*) ; b=$(ls "$work"/b.*)

echo "== stitch"
bash "$here/stitch.sh" "$work/out/audio.wav" "$a" "$b" | tee "$work/stitch.txt"
[[ -s "$work/out/audio.wav" ]] || fail "no stitched wav"
[[ $(wc -l < "$work/out/audio.chapters.txt") -eq 2 ]] || fail "chapter map should have 2 rows"
grep -q '^00:00:00' "$work/out/audio.chapters.txt" || fail "first chapter must start at 0"
tail -1 "$work/out/audio.chapters.txt" | grep -qv '^00:00:00' || fail "second chapter offset not advanced"

echo "== glossary packing"
"$PY" - <<'PY' || fail "build_prompt"
import sys
sys.path.insert(0, __import__("os").environ["SANAR_SCRIPTS"])
from transcribe import build_prompt, PROMPT_TOKEN_BUDGET, approx_len
p, used, dropped = build_prompt([], approx_len)
assert p is None and used == 0
terms = [f"Term{i}" for i in range(5000)]
p, used, dropped = build_prompt(terms, approx_len)
assert approx_len(p) <= PROMPT_TOKEN_BUDGET, "prompt overflows the decoder window"
assert used + dropped == len(terms) and dropped > 0, (used, dropped)
assert p.startswith("Term0, Term1"), "ranking order must be preserved"
print(f"ok: packed {used}, dropped {dropped}")
PY

echo "== transcribe (tiny)"
printf 'ClickHouse\nKubernetes\npull request\n' > "$work/terms.txt"
model=tiny   # valid for both backends; transcribe.py maps it to mlx-community/whisper-tiny-mlx
SANAR_MODEL="$model" "$PY" "$here/transcribe.py" \
  --audio "$work/out/audio.wav" --lang en \
  --terms "$work/terms.txt" --out "$work/out/pass-en.txt" | tee "$work/pass.txt"
grep -qE '^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\] ' "$work/out/pass-en.txt" || fail "no timestamped segments"
"$PY" -c "
import json,sys
m=json.load(open('$work/out/pass-en.txt.json'))
assert m['prompt_terms_used']==3, m
assert m['segments']>0, m
assert m['audio_seconds']>0, m
print('ok:', m['backend'], m['device'], m['segments'], 'segments')
" || fail "metadata"

echo
echo "PASS — all phases wired (model=$model, backend=$backend)"
