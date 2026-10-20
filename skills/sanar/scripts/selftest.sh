#!/usr/bin/env bash
# End-to-end smoke test with whisper-tiny (~75 MB) instead of large-v3 (~3 GB).
# Proves: probe, setup, stitch offsets, diarization + voice match, glossary packing, chunked
# multi-language transcription, output format.
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
# Two voices, so diarization and the chapter map have something to prove, plus a separate
# clip of the first voice to enroll. `say` on macOS, espeak-ng elsewhere.
mkdir -p "$work/voices"
A="Мы задеплоили ClickHouse в Kubernetes, и ночью кластер упал."
B="We merged the pull request and deployed it to production yesterday."
V="Это образец моего голоса для проверки идентификации."
if command -v say >/dev/null; then
  say -v Milena -o "$work/a.aiff" "$A"; say -v Daniel -o "$work/b.aiff" "$B"
  say -v Milena -o "$work/voices/Milena.aiff" "$V"
elif command -v espeak-ng >/dev/null; then
  espeak-ng -v ru -w "$work/a.wav" "$A"; espeak-ng -v en-us -w "$work/b.wav" "$B"
  espeak-ng -v ru -w "$work/voices/Milena.wav" "$V"
else
  speech=none
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
from transcribe import build_prompt, PROMPT_TOKEN_BUDGET, approx_len, pack, MAX_CHUNK
p, used, dropped = build_prompt([], approx_len)
assert p is None and used == 0
terms = [f"Term{i}" for i in range(5000)]
p, used, dropped = build_prompt(terms, approx_len)
assert approx_len(p) <= PROMPT_TOKEN_BUDGET, "prompt overflows the decoder window"
assert used + dropped == len(terms) and dropped > 0, (used, dropped)
assert p.startswith("Term0, Term1"), "ranking order must be preserved"
turns = [[0, 5, "S1"], [5.2, 9, "S2"], [9.1, 40, "S1"], [45, 47, "S2"]]
chunks = pack(turns)
assert all(e - s <= MAX_CHUNK for s, e in chunks), chunks
assert chunks[0] == [0, 9], "short turns must share a chunk"
assert len(chunks) == 4 and abs((chunks[1][1] - chunks[1][0]) - 15.45) < 1e-6, \
    "a turn over MAX_CHUNK splits into equal parts"
assert chunks[-1] == [45, 47], "a long pause must start a new chunk"
print(f"ok: packed {used}, dropped {dropped}; {len(chunks)} chunks")
PY

if [[ "${speech:-}" == none ]]; then
  echo
  echo "PARTIAL PASS — no say/espeak-ng to synthesize speech, diarize and transcribe untested"
  exit 0
fi

echo "== diarize"
"$PY" "$here/diarize.py" --audio "$work/out/audio.wav" --out "$work/out/speakers.json" \
  --voices "$work/voices" | tee "$work/diarize.txt"
"$PY" -c "
import json
d=json.load(open('$work/out/speakers.json'))
assert d['turns'] and d['speakers'], d
named=[s for s in d['speakers'] if (s['voice'] or {}).get('name')=='Milena']
assert named, ('enrolled voice not matched', d['speakers'])
print('ok:', len(d['speakers']), 'speakers;', named[0]['id'], '= Milena', named[0]['voice'])
" || fail "speakers.json"

echo "== transcribe (tiny)"
printf 'ClickHouse\nKubernetes\npull request\n' > "$work/terms.txt"
SANAR_MODEL=tiny "$PY" "$here/transcribe.py" \
  --audio "$work/out/audio.wav" --speakers "$work/out/speakers.json" --lang ru,en \
  --terms "$work/terms.txt" --out-dir "$work/out" | tee "$work/pass.txt"
for l in ru en; do
  grep -q '^@c001 ' "$work/out/pass-$l.txt" || fail "pass-$l: no chunk header"
  grep -qE '^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\] S[0-9]+: ' "$work/out/pass-$l.txt" \
    || fail "pass-$l: no speaker-labelled lines"
done
"$PY" -c "
import json
m=json.load(open('$work/out/pass-en.txt.json'))
assert m['prompt_terms_used']==3, m
assert m['chunks']>0 and m['audio_seconds']>0, m
print('ok:', m['backend'], m['device'], m['chunks'], 'chunks, rtf', m['realtime_factor'])
" || fail "metadata"

echo
echo "PASS — all phases wired (model=tiny, backend=$backend)"
