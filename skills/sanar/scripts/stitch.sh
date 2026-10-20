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

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
map="${out%.*}.chapters.txt"
: > "$map"
off=0
for i in "${!files[@]}"; do
  # Decode each source on its own and take the offset from the decoded PCM: a container's
  # duration can belong to a longer video stream and would shift every later chapter.
  ffmpeg -nostdin -y -loglevel error -i "${files[$i]}" -vn -ar 16000 -ac 1 -c:a pcm_s16le "$tmp/$i.wav"
  printf '%s\t%s\n' "$(ts "$off")" "${files[$i]}" >> "$map"
  off=$(awk -v a="$off" -v b="$(dur "$tmp/$i.wav")" 'BEGIN{print a+b}')
  printf "file '%s'\n" "$tmp/$i.wav" >> "$tmp/list.txt"
done
ffmpeg -nostdin -y -loglevel error -f concat -safe 0 -i "$tmp/list.txt" -c copy "$out"

total=$(dur "$out")
printf 'out=%s\nduration=%s\nduration_hms=%s\nsources=%d\nchapters=%s\n' \
  "$out" "$total" "$(ts "$total")" "${#files[@]}" "$map"
cat "$map"
