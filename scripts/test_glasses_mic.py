#!/usr/bin/env python3
"""Quick check: glasses mic via RTMP audio (preferred) or BLE PCM on port 8767."""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mic_ingest import MicIngestServer, glasses_mic_buffer  # noqa: E402
from src.rtmp_audio_ingest import RtmpAudioIngest  # noqa: E402

SAMPLE_RATE = 16000
MIC_PORT = 8767
MIC_HEALTH = f"http://127.0.0.1:{MIC_PORT}/health"
DEFAULT_RTMP = "rtmp://127.0.0.1:1935/live/mentra-live"


def rms_level(pcm: bytes) -> float:
    if not pcm or len(pcm) < 2:
        return 0.0
    total = 0
    count = 0
    for i in range(0, len(pcm) - 1, 2):
        sample = int.from_bytes(pcm[i : i + 2], "little", signed=True)
        total += sample * sample
        count += 1
    if count == 0:
        return 0.0
    return math.sqrt(total / count) / 32768.0


def fetch_health() -> dict | None:
    try:
        with urllib.request.urlopen(MIC_HEALTH, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def monitor_loop(read_local_buffer: bool):
    last_total = 0
    started_at = time.monotonic()
    while True:
        if read_local_buffer:
            buf = glasses_mic_buffer()
            total = buf.total_bytes
            live = buf.is_live(within_seconds=2.0)
        else:
            payload = fetch_health()
            if payload is None:
                print("  mic server on 8767 not reachable")
                time.sleep(1.0)
                continue
            total = int(payload.get("bytes") or 0)
            live = bool(payload.get("live"))

        delta = total - last_total
        last_total = total

        if live:
            bar = "#" * min(40, int(delta / 800))
            print(f"  LIVE  +{delta:5d} bytes  total={total:7d}  {bar}")
        elif total > 0:
            print(f"  idle  (no new audio for 2s)  total={total}")
        else:
            elapsed = int(time.monotonic() - started_at)
            print(f"  waiting... ({elapsed}s) — no audio yet", end="\r")

        time.sleep(1.0)


def main_rtmp(rtmp_url: str):
    print(f"Glasses mic via RTMP audio: {rtmp_url}")
    print("Prerequisites:")
    print("  1. ./scripts/run_rtmp_server.sh  (mediamtx on Mac)")
    print("  2. Phone: Trail Return Lab -> Glasses -> Start everything")
    print("  3. Speak into the GLASSES (not the Mac). Ctrl+C to stop.\n")
    ingest = RtmpAudioIngest(rtmp_url).start()
    try:
        monitor_loop(read_local_buffer=True)
    finally:
        ingest.stop()


def main_ble():
    MicIngestServer.verbose = True
    server = None
    read_local = True

    if MicIngestServer.port_in_use(MIC_PORT):
        payload = fetch_health()
        if payload and payload.get("ok"):
            print(f"Port {MIC_PORT} already in use — monitoring existing mic server")
            print(f"Health: {payload}")
            read_local = False
        else:
            print(f"Port {MIC_PORT} is busy but not our mic server.")
            print("Free it with: lsof -ti :8767 | xargs kill")
            raise SystemExit(1)
    else:
        server = MicIngestServer(port=MIC_PORT).start()
        print(f"Listening for glasses mic (BLE PCM) on http://0.0.0.0:{MIC_PORT}/mic/pcm")

    print("On phone: Trail Return Lab -> Glasses -> Start everything")
    print("Then speak into the GLASSES (not the Mac). Ctrl+C to stop.\n")
    try:
        monitor_loop(read_local)
    finally:
        if server is not None:
            server.stop()


def record_sample(seconds: float = 4.0, out_path: str = "glasses_mic_test.wav", rtmp_url: str | None = None):
    server = None
    ingest = None
    if rtmp_url:
        ingest = RtmpAudioIngest(rtmp_url).start()
        print(f"Recording {seconds:.0f}s from RTMP glasses mic: {rtmp_url}")
    else:
        if MicIngestServer.port_in_use(MIC_PORT) and fetch_health() is None:
            print(f"Port {MIC_PORT} busy. Free it: lsof -ti :8767 | xargs kill")
            return 1
        if not MicIngestServer.port_in_use(MIC_PORT):
            server = MicIngestServer(port=MIC_PORT).start()
        print(f"Recording {seconds:.0f}s from BLE glasses mic on port {MIC_PORT}")

    print("Phone: Glasses -> Start everything, then speak.\n")

    pcm = glasses_mic_buffer().read_seconds(seconds, timeout=seconds + 20)
    if not pcm:
        print("FAIL: no audio received. Check phone app + same Wi-Fi + RTMP stream has audio track.")
        if server is not None:
            server.stop()
        if ingest is not None:
            ingest.stop()
        return 1

    out = Path(out_path)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)

    level = rms_level(pcm)
    print(f"OK: saved {out} ({len(pcm)} bytes, level={level:.2f})")
    print(f"Play back: afplay {out}")
    if server is not None:
        server.stop()
    if ingest is not None:
        ingest.stop()
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rtmp":
        url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_RTMP
        try:
            main_rtmp(url)
        except KeyboardInterrupt:
            print("\nStopped.")
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        secs = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
        rtmp = sys.argv[3] if len(sys.argv) > 3 else None
        raise SystemExit(record_sample(secs, rtmp_url=rtmp))
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        print(json.dumps(fetch_health(), indent=2))
        raise SystemExit(0)
    try:
        main_rtmp(DEFAULT_RTMP)
    except KeyboardInterrupt:
        print("\nStopped.")
