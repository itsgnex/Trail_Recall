#!/usr/bin/env bash
set -euo pipefail

export ANDROID_TRAIL_URL="${ANDROID_TRAIL_URL:-http://10.117.240.233:8766}"
python main.py --camera-source mentra
