#!/usr/bin/env bash
# Builds the short live-check clip: a spoken line (macOS `say`) over a plain
# background, with "Paid partnership with Dasho" on screen from 1 s to 3 s.
# Homebrew ffmpeg ships without the drawtext filter, so the caption is
# rendered to a PNG with headless Chrome and overlaid with ffmpeg.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:-live-check.mp4}
WORK=$(mktemp -d)
trap 'find "$WORK" -delete' EXIT
LINE="I tried Dasho for our brand posts. It guarantees zero FDA violations, one hundred percent. Every draft gets a named approver before it goes live."
say -v Samantha -o "$WORK/voice.aiff" "$LINE"
cat > "$WORK/caption.html" <<'HTML'
<html><body style="margin:0;background:transparent;display:flex;align-items:flex-end;justify-content:center;height:720px;width:1280px">
<div style="margin-bottom:90px;font:600 56px Helvetica,Arial,sans-serif;color:#fff;background:#111;padding:18px 34px;border-radius:12px">Paid partnership with Dasho</div>
</body></html>
HTML
CHROME=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --default-background-color=00000000 \
  --window-size=1280,720 --screenshot="$WORK/caption.png" "file://$WORK/caption.html" >/dev/null 2>&1
DUR=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$WORK/voice.aiff")
ffmpeg -v error -y -f lavfi -i "color=c=0x2F6F62:s=1280x720:d=$DUR:r=25" -i "$WORK/caption.png" -i "$WORK/voice.aiff" \
  -filter_complex "[0][1]overlay=0:0:enable='between(t,1,3)'[v]" -map "[v]" -map 2:a \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 96k -shortest "$OUT"
echo "wrote $OUT ($DUR s)"
