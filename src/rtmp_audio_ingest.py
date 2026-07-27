from __future__ import annotations

import shutil
import subprocess
import threading
import time
from urllib.parse import urlparse

from . import app_log


def _local_rtsp_url(stream_url: str) -> str:
    """Read audio from the local mediamtx RTSP stream."""
    parsed = urlparse(stream_url)
    path = parsed.path or "/live/mentra-live"
    return f"rtsp://127.0.0.1:8554{path}"


class RtmpAudioIngest:
    """Pull glasses mic audio from mediamtx RTSP and append 16 kHz mono PCM."""

    def __init__(self, rtmp_url: str):
        self.rtmp_url = rtmp_url
        self.audio_url = _local_rtsp_url(rtmp_url)
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._bytes = 0
        self.started = False
        self._reconnecting = False
        self._attempt = 0

    @property
    def total_bytes(self) -> int:
        return self._bytes

    def start(self):
        if self._thread and self._thread.is_alive():
            return self
        if not shutil.which("ffmpeg"):
            print("Glasses mic RTSP: ffmpeg not found — falling back to HTTP PCM ingest")
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.started = True
        print(f"Glasses mic via RTSP audio: {self.audio_url}")
        return self

    def stop(self):
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _run(self):
        from .mic_ingest import glasses_mic_buffer

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "+nobuffer+genpts+igndts",
            "-flags",
            "low_delay",
            "-use_wallclock_as_timestamps",
            "1",
            "-i",
            self.audio_url,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            "highpass=f=140,afftdn,aresample=async=1:first_pts=0,asetpts=N/SR/TB",
            "-acodec",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ]
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                assert self._proc.stdout is not None
                while not self._stop.is_set():
                    chunk = self._proc.stdout.read(4096)
                    if not chunk:
                        break
                    if self._reconnecting:
                        print("STREAM connected")
                        self._reconnecting = False
                        self._attempt = 0
                    glasses_mic_buffer().append(chunk, mode="rtsp")
                    self._bytes += len(chunk)
                code = self._proc.wait(timeout=5)
                if code != 0 and not self._stop.is_set():
                    err = (self._proc.stderr.read() if self._proc.stderr else b"").decode("utf-8", errors="replace")
                    if err.strip():
                        app_log.debug(f"Glasses mic RTSP ffmpeg exited: {err.strip()[:500]}")
            except Exception as exc:
                if not self._stop.is_set():
                    print(f"Glasses mic RTSP: {exc}")
            finally:
                self._proc = None
            if not self._stop.is_set():
                if not self._reconnecting:
                    print("STREAM disconnected reason=publisher_stopped")
                    try:
                        from .speech_out import cancel_reply_capture

                        cancel_reply_capture("stream_disconnect")
                        glasses_mic_buffer().clear()
                    except Exception:
                        pass
                    self._reconnecting = True
                self._attempt += 1
                delay = min(8, 2 ** min(self._attempt - 1, 3))
                print(f"STREAM reconnecting attempt={self._attempt} nextRetrySeconds={delay}")
                time.sleep(delay)
