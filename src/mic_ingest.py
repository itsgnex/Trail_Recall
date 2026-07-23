from __future__ import annotations

import threading
import time
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
MAX_BUFFER_SECONDS = 30


class GlassesMicBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._pcm = bytearray()
        self._last_at = 0.0
        self._total_bytes = 0
        self._last_log_at = 0.0
        self._last_log_bytes = 0
        self._mode = "http_pcm"

    def append(self, data: bytes, mode: str = "http_pcm"):
        if not data:
            return
        max_bytes = SAMPLE_RATE * SAMPLE_WIDTH * MAX_BUFFER_SECONDS
        with self._lock:
            self._mode = mode
            self._pcm.extend(data)
            if len(self._pcm) > max_bytes:
                self._pcm = self._pcm[-max_bytes:]
            self._last_at = time.monotonic()
            self._total_bytes += len(data)
            interval = float(os.getenv("MIC_LOG_INTERVAL_SECONDS", "10"))
            if self._last_at - self._last_log_at >= interval:
                elapsed = max(0.001, self._last_at - self._last_log_at) if self._last_log_at else 5.0
                rate = int((self._total_bytes - self._last_log_bytes) / elapsed)
                if os.getenv("LOG_LEVEL", "INFO").strip().upper() in {"DEBUG", "TRACE"}:
                    print(f"MIC_SOURCE mode={self._mode} bytesPerSecond={rate} totalBytes={self._total_bytes}")
                self._last_log_at = self._last_at
                self._last_log_bytes = self._total_bytes

    def clear(self):
        with self._lock:
            self._pcm.clear()

    def read_seconds(self, seconds: float, timeout: float = 12.0) -> bytes | None:
        need = int(SAMPLE_RATE * SAMPLE_WIDTH * max(0.5, seconds))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self._pcm) >= need:
                    chunk = bytes(self._pcm[:need])
                    del self._pcm[:need]
                    return chunk
            time.sleep(0.05)
        with self._lock:
            min_bytes = int(SAMPLE_RATE * SAMPLE_WIDTH * 0.5)
            if len(self._pcm) >= min_bytes:
                chunk = bytes(self._pcm)
                self._pcm.clear()
                return chunk
        return None

    def read_bytes(self, need: int, timeout: float = 12.0) -> bytes | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self._pcm) >= need:
                    chunk = bytes(self._pcm[:need])
                    del self._pcm[:need]
                    return chunk
            time.sleep(0.02)
        return None

    def read_fresh_seconds(self, seconds: float, timeout: float = 12.0) -> bytes | None:
        self.clear()
        return self.read_seconds(seconds, timeout=timeout)

    def is_live(self, within_seconds: float = 2.0) -> bool:
        with self._lock:
            return self._last_at > 0 and (time.monotonic() - self._last_at) <= within_seconds

    @property
    def total_bytes(self) -> int:
        return self._total_bytes


_BUFFER = GlassesMicBuffer()


def glasses_mic_buffer() -> GlassesMicBuffer:
    return _BUFFER


class _MicHandler(BaseHTTPRequestHandler):
    server_version = "MentraMicIngest/1.0"

    def do_GET(self):
        if self.path in {"/", "/health"}:
            live = _BUFFER.is_live()
            body = f'{{"ok":true,"live":{str(live).lower()},"bytes":{_BUFFER.total_bytes}}}'.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/mic/pcm":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        data = self.rfile.read(length) if length > 0 else b""
        _BUFFER.append(data)
        if getattr(MicIngestServer, "verbose", False) and data:
            print(f"  POST +{len(data)} bytes (total={_BUFFER.total_bytes})")
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        return


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class MicIngestServer:
    verbose = False

    def __init__(self, host: str = "0.0.0.0", port: int = 8767):
        self.host = host
        self.port = port
        try:
            self._server = _ReusableThreadingHTTPServer((host, port), _MicHandler)
        except OSError as exc:
            raise SystemExit(f"MIC_SERVER\nstatus=FAILED\nendpoint={host}:{port}\nerror={exc}") from exc
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @staticmethod
    def port_in_use(port: int = 8767, host: str = "127.0.0.1") -> bool:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0

    def start(self):
        self._thread.start()
        print(f"MIC_SERVER\nstatus=LISTENING\nendpoint={self.host}:{self.port}/mic/pcm")
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
