#!/usr/bin/env bash
set -euo pipefail

export ANDROID_TRAIL_URL="${ANDROID_TRAIL_URL:-http://192.168.0.91:8766}"
python main.py --camera-source mentra
