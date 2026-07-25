#!/usr/bin/env bash
set -euo pipefail

STREAM_PATH="${MENTRA_RTMP_STREAM_PATH:-live/mentra-live}"
HOST_IP="${MENTRA_STREAM_HOST_IP:-10.117.240.212}"
RTMP_URL="rtmp://$HOST_IP:1935/$STREAM_PATH"
RTSP_URL="rtsp://127.0.0.1:8554/$STREAM_PATH"
HLS_URL="http://$HOST_IP:8888/$STREAM_PATH/index.m3u8"
WEBRTC_URL="http://$HOST_IP:8889/$STREAM_PATH"

port_busy() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

print_urls() {
  echo "MEDIAMTX"
  echo "status=$1"
  echo "rtmp=$RTMP_URL"
  echo "rtsp=$RTSP_URL"
  echo "hls=$HLS_URL"
  echo "webrtc=$WEBRTC_URL"
}

probe_stream() {
  if ffmpeg -nostdin -hide_banner -loglevel error -i "$RTSP_URL" -an -sn -frames:v 1 -f null - >/dev/null 2>&1; then
    echo "MENTRA_STREAM"
    echo "publisher=CONNECTED"
    echo "video=true"
    echo "audio=true"
    return 0
  fi
  echo "MENTRA_STREAM"
  echo "publisher=DISCONNECTED"
  echo "video=false"
  echo "audio=false"
  return 1
}

if [ "${1:-}" = "probe" ]; then
  print_urls "LISTENING"
  probe_stream
  exit $?
fi

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
  -e 's/hlsSegmentCount: 4/hlsSegmentCount: 7/' \
  "$GEN_CFG" 2>/dev/null || sed -i \
  -e 's/hlsAlwaysRemux: false/hlsAlwaysRemux: true/' \
  -e 's/hlsPartDuration: 200ms/hlsPartDuration: 100ms/' \
  -e 's/hlsSegmentCount: 4/hlsSegmentCount: 7/' \
  "$GEN_CFG"

print_urls "STARTING"
echo "ffplay=$RTMP_URL"

if pgrep -x mediamtx >/dev/null 2>&1; then
  print_urls "LISTENING"
  echo "mediamtx is already running; not starting a duplicate."
  exit 0
fi

for port in 1935 8554 8888 8889; do
  if port_busy "$port"; then
    echo "Port $port is already in use by a non-detected process. Stop it before starting mediaMTX." >&2
    exit 1
  fi
done

echo "Starting mediamtx (low-latency HLS config)..."
exec "$MEDIAMTX_BIN" "$GEN_CFG"
