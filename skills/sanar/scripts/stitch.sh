#!/usr/bin/env bash
# Concat N audio/video files into one 16 kHz mono WAV, plus a chapter map of source offsets.
# usage: stitch.sh OUT.wav IN1 [IN2 ...]
set -euo pipefail

[[ $# -ge 2 ]] || { echo "usage: stitch.sh OUT.wav IN1 [IN2 ...]" >&2; exit 2; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }

out=$1; shift
files=("$@")
mkdir -p "$(dirname "$out")"

for f in "${files[@]}"; do
  [[ -r "$f" ]] || { echo "unreadable: $f" >&2; exit 1; }
done

dur() { ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$1" 2>/dev/null || echo 0; }
ts() { awk -v s="$1" 'BEGIN{printf "%02d:%02d:%02d", int(s/3600), int(s%3600/60), int(s%60)}'; }

if [[ ${#files[@]} -eq 1 ]]; then
  ffmpeg -nostdin -y -loglevel error -i "${files[0]}" \
    -vn -ar 16000 -ac 1 -c:a pcm_s16le "$out"
else
  # concat filter, not the concat demuxer: it re-decodes, so mixed containers/codecs are fine.
  args=(); filt=""
  for i in "${!files[@]}"; do args+=(-i "${files[$i]}"); filt+="[$i:a]"; done
  ffmpeg -nostdin -y -loglevel error "${args[@]}" \
    -filter_complex "${filt}concat=n=${#files[@]}:v=0:a=1[a]" -map '[a]' \
    -ar 16000 -ac 1 -c:a pcm_s16le "$out"
fi

map="${out%.*}.chapters.txt"
: > "$map"
off=0
for f in "${files[@]}"; do
  printf '%s\t%s\n' "$(ts "$off")" "$f" >> "$map"
  off=$(awk -v a="$off" -v b="$(dur "$f")" 'BEGIN{print a+b}')
done

total=$(dur "$out")
printf 'out=%s\nduration=%s\nduration_hms=%s\nsources=%d\nchapters=%s\n' \
  "$out" "$total" "$(ts "$total")" "${#files[@]}" "$map"
cat "$map"
