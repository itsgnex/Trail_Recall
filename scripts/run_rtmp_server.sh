#!/usr/bin/env bash
set -euo pipefail

STREAM_PATH="${MENTRA_RTMP_STREAM_PATH:-live/mentra-live}"
HOST_IP="${MENTRA_STREAM_HOST_IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)}"

if [ -z "$HOST_IP" ]; then
  echo "Set MENTRA_STREAM_HOST_IP to your Mac LAN IP (for example 192.168.1.42)." >&2
  exit 1
fi

MEDIAMTX_BIN="${MEDIAMTX_BIN:-$(command -v mediamtx || true)}"
BASE_CFG="${MEDIAMTX_CFG:-/opt/homebrew/etc/mediamtx/mediamtx.yml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_CFG="${SCRIPT_DIR}/.mediamtx-mentra.generated.yml"

if [ -z "$MEDIAMTX_BIN" ]; then
  echo "mediamtx is not installed. Run: brew install mediamtx" >&2
  exit 1
fi

if [ ! -f "$BASE_CFG" ]; then
  echo "mediamtx base config not found at $BASE_CFG" >&2
  exit 1
fi

# Low-latency HLS for browser preview (default brew config buffers ~3–5s).
cp "$BASE_CFG" "$GEN_CFG"
sed -i '' \
  -e 's/hlsAlwaysRemux: false/hlsAlwaysRemux: true/' \
  -e 's/hlsPartDuration: 200ms/hlsPartDuration: 100ms/' \
  -e 's/hlsSegmentCount: 7/hlsSegmentCount: 4/' \
  "$GEN_CFG" 2>/dev/null || sed -i \
  -e 's/hlsAlwaysRemux: false/hlsAlwaysRemux: true/' \
  -e 's/hlsPartDuration: 200ms/hlsPartDuration: 100ms/' \
  -e 's/hlsSegmentCount: 7/hlsSegmentCount: 4/' \
  "$GEN_CFG"

echo "RTMP publish URL: rtmp://$HOST_IP:1935/$STREAM_PATH"
echo "HLS preview URL:  http://$HOST_IP:8888/$STREAM_PATH  (~1–2s with low-latency settings)"
echo "WebRTC preview:   http://$HOST_IP:8889/$STREAM_PATH  (near real-time — use this to check latency)"
echo "ffplay (RTMP):      ffplay -fflags nobuffer -flags low_delay -framedrop rtmp://$HOST_IP:1935/$STREAM_PATH"

if lsof -nP -iTCP:1935 -sTCP:LISTEN >/dev/null 2>&1; then
  echo ""
  echo "mediamtx is already running on port 1935."
  echo "Restart to apply low-latency HLS settings:"
  echo "  pkill mediamtx && sleep 1 && $0"
  exit 0
fi

echo "Starting mediamtx (low-latency HLS config)..."
exec "$MEDIAMTX_BIN" "$GEN_CFG"
