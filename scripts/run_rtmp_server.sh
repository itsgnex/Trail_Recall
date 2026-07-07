#!/usr/bin/env bash
set -euo pipefail

STREAM_PATH="${MENTRA_RTMP_STREAM_PATH:-live/mentra-live}"
HOST_IP="${MENTRA_STREAM_HOST_IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)}"

if [ -z "$HOST_IP" ]; then
  echo "Set MENTRA_STREAM_HOST_IP to your Mac LAN IP (for example 192.168.1.42)." >&2
  exit 1
fi

MEDIAMTX_BIN="${MEDIAMTX_BIN:-$(command -v mediamtx || true)}"
MEDIAMTX_CFG="${MEDIAMTX_CFG:-/opt/homebrew/etc/mediamtx/mediamtx.yml}"

if [ -z "$MEDIAMTX_BIN" ]; then
  echo "mediamtx is not installed. Run: brew install mediamtx" >&2
  exit 1
fi

echo "RTMP publish URL: rtmp://$HOST_IP:1935/$STREAM_PATH"
echo "HLS preview URL:  http://$HOST_IP:8888/$STREAM_PATH"
echo "ffplay preview:     ffplay -fflags nobuffer -flags low_delay -framedrop rtmp://$HOST_IP:1935/$STREAM_PATH"

echo "Starting mediamtx..."
exec "$MEDIAMTX_BIN" "$MEDIAMTX_CFG"
