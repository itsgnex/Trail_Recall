#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export MENTRA_STREAM_HOST_IP="${MENTRA_STREAM_HOST_IP:-10.117.240.212}"
export ANDROID_TRAIL_URL="http://10.117.240.233:8766"
PHONE_IP="10.117.240.233"
STREAM_URL="rtmp://${MENTRA_STREAM_HOST_IP}:1935/live/mentra-live"

mac_ip="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
if [ "$mac_ip" != "$MENTRA_STREAM_HOST_IP" ]; then
  echo "NETWORK_CHECK"
  echo "status=FAILED"
  echo "expectedMacIp=$MENTRA_STREAM_HOST_IP"
  echo "actualMacIp=${mac_ip:-UNKNOWN}"
  echo "Reconnect to the phone hotspot or update MENTRA_STREAM_HOST_IP."
  exit 1
fi

if ! curl --connect-timeout 3 --max-time 5 -fsS "$ANDROID_TRAIL_URL/health" >/dev/null; then
  echo "ANDROID_BRIDGE"
  echo "status=UNAVAILABLE"
  echo "url=$ANDROID_TRAIL_URL"
  echo "Confirm the phone hotspot and Android navigation server are running."
  exit 1
fi

echo "ANDROID_BRIDGE"
echo "status=CONNECTED"
echo "url=$ANDROID_TRAIL_URL"

if ! lsof -nP -iTCP:1935 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Starting MediaMTX..."
  ./scripts/run_rtmp_server.sh > /tmp/metra-live-mediamtx.log 2>&1 &
  sleep 2
else
  ./scripts/run_rtmp_server.sh
fi

for port in 1935 8554 8888 8889; do
  if ! lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "MEDIAMTX"
    echo "status=FAILED"
    echo "missingPort=$port"
    echo "See /tmp/metra-live-mediamtx.log"
    exit 1
  fi
done

echo "Waiting for Android RTMP publisher: $STREAM_URL"
deadline=$((SECONDS + 120))
until ./scripts/run_rtmp_server.sh probe >/tmp/metra-live-probe.log 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    cat /tmp/metra-live-probe.log
    echo "Stream publisher never appeared. On phone: Trail Return Lab -> Glasses -> Start everything."
    exit 1
  fi
  sleep 2
done
cat /tmp/metra-live-probe.log

python main.py --camera-source mentra
