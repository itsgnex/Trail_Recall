import array
import fcntl
import os
import re
import subprocess
import threading
import time
import termios
from pathlib import Path

import cv2
import numpy as np

from . import app_log


STREAM_SCHEMES = ("http://", "https://", "rtmp://", "rtsp://", "udp://", "srt://")
DEFAULT_STREAM_WIDTH = 1280
DEFAULT_STREAM_HEIGHT = 720
MAX_STALE_FRAME_SECONDS = 0.5
STREAM_GLITCH_HOLD_SECONDS = 0.5
OPENCV_STREAM_STALE_SECONDS = 1.5
OPENCV_STREAM_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
DEFAULT_RTMP_CAPTURE_OPTIONS = "rtmp_live;live|rw_timeout;8000000"
OPENCV_FRAME_DRAIN = int(os.getenv("MENTRA_STREAM_FRAME_DRAIN", "2"))


def discover_cameras(max_index=10):
    cameras = []
    old_log_level = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        for index in range(max_index + 1):
            capture = cv2.VideoCapture(index)
            ok, frame = capture.read() if capture.isOpened() else (False, None)
            capture.release()
            if not ok:
                continue

            cameras.append(index)
            print(f"Camera {index} is available.")
            cv2.putText(frame, f"Camera ID: {index}", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(f"Camera ID: {index}", frame)
            cv2.waitKey(700)
            cv2.destroyWindow(f"Camera ID: {index}")
    finally:
        os.dup2(stderr, 2)
        os.close(stderr)
        os.close(devnull)
        if old_log_level is not None:
            cv2.setLogLevel(old_log_level)
    return cameras


def choose_camera(max_index=10):
    cameras = discover_cameras(max_index)
    if not cameras:
        print("No cameras found from index 0 to 10. Check camera connections and macOS camera permission.")
        return None

    while True:
        choice = input(f"Choose camera {cameras}: ").strip()
        try:
            index = int(choice)
        except ValueError:
            print("Please type a camera number.")
            continue
        if index in cameras:
            return index
        print(f"Camera {index} was not available. Choose one of {cameras}.")


def _is_stream_source(source):
    return isinstance(source, str) and source.lower().startswith(STREAM_SCHEMES)


def _is_hls_source(source):
    lower = (source or "").lower()
    return lower.startswith(("http://", "https://")) and (".m3u8" in lower or "/live/" in lower)


def resolve_ingest_url(source):
    """Prefer HLS for mediamtx RTMP URLs — Mac ffmpeg RTMP pipe ingest is often flaky."""
    _, ingest = split_stream_urls(source)
    return ingest


def split_stream_urls(source):
    """Return (publish_wait_url, ingest_url). Default: RTMP ingest (low latency). Set MENTRA_USE_HLS_INGEST=1 for HLS."""
    if not isinstance(source, str):
        return source, source
    text = source.strip()
    match = re.match(r"rtmp://([^/]+)/(.+)", text, re.IGNORECASE)
    if not match:
        return text, text
    rtmp = text
    use_hls = os.getenv("MENTRA_USE_HLS_INGEST", "").strip().lower() in {"1", "true", "yes"}
    if not use_hls:
        print(f"Stream ingest: RTMP ({rtmp})")
        return rtmp, rtmp
    host_port, path = match.group(1), match.group(2).rstrip("/")
    host = host_port.split(":")[0]
    hls = f"http://{host}:8888/{path}/index.m3u8"
    print(f"Stream ingest: HLS ({hls}) — set MENTRA_USE_HLS_INGEST=0 to use RTMP instead")
    print(f"Stream wait: probing RTMP publisher at {rtmp}")
    return rtmp, hls


def _ffmpeg_input_args(source):
    args = []
    lower = source.lower()
    if lower.startswith("rtmp://") or lower.startswith("rtmps://"):
        args.extend(["-rw_timeout", "5000000", "-rtmp_live", "live"])
    if lower.startswith(("http://", "https://")):
        args.extend(
            [
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                "2",
            ]
        )
        if _is_hls_source(source):
            args.extend(
                [
                    "-live_start_index",
                    "-1",
                ]
            )
    return args


def wait_for_publisher(source, max_wait_seconds=180, poll_seconds=3, ingest_source=None):
    """Block until the stream publisher is live (RTMP probe), optionally until HLS is readable."""
    if not _is_stream_source(source):
        return True

    print(f"Waiting for glasses stream at {source!r}")
    print("On phone: Trail Return Lab -> Glasses -> Start everything (connect glasses first)")
    deadline = time.monotonic() + max_wait_seconds
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        if _probe_stream_frame(source):
            print("MENTRA_STREAM\npublisher=CONNECTED\nvideo=true\naudio=true")
            if ingest_source and ingest_source != source:
                for hls_try in range(12):
                    if _probe_stream_frame(ingest_source, timeout_seconds=6):
                        print("Stream publisher is live (HLS ready).")
                        return True
                    time.sleep(0.5)
                print("Stream publisher is live (RTMP up; HLS may need a few seconds).")
                return True
            print("Stream publisher is live.")
            return True
        remaining = int(deadline - time.monotonic())
        print(f"  no stream yet (attempt {attempt}, ~{remaining}s left) — start publishing on the phone")
        time.sleep(poll_seconds)

    print(
        "Timed out waiting for stream. Checklist:\n"
        "  1. Terminal 1: ./scripts/run_rtmp_server.sh is running (only one mediamtx instance)\n"
        "  2. Phone app: Glasses -> Start everything (RTMP publishing)\n"
        "  3. Phone and Mac on same Wi-Fi\n"
        "  4. Stream URL on phone matches rtmp://YOUR_MAC_IP:1935/live/mentra-live"
    )
    return False


def _probe_stream_frame(source, timeout_seconds=8):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *_ffmpeg_input_args(source),
        "-i",
        source,
        "-an",
        "-sn",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout_seconds)
        return result.returncode == 0
    except Exception:
        return False


class _LocalCamera:
    def __init__(self, source):
        self.source = source
        self.capture = None
        self.opened = False

    def __enter__(self):
        self.capture = cv2.VideoCapture(self.source)
        self.opened = self.capture.isOpened()
        return self

    def __exit__(self, *_):
        if self.capture:
            self.capture.release()

    def read(self):
        for _ in range(3):
            ok, frame = self.capture.read()
            if ok:
                return frame
        return None

    def show(self, frame):
        cv2.imshow("Gaze Assistant", frame)
        return cv2.waitKey(1) & 0xFF == ord("q")


class _OpenCvStreamCamera:
    """Read network streams with cv2.VideoCapture — smooth RTMP ingest with light buffering."""

    def __init__(self, source, width=DEFAULT_STREAM_WIDTH, height=DEFAULT_STREAM_HEIGHT):
        self.source = source
        self.width = width
        self.height = height
        self.capture = None
        self.opened = False
        self._last_frame = None
        self._last_frame_at = 0.0
        self._miss_count = 0
        self._reconnecting = False
        self._reconnect_attempt = 0
        self._next_reconnect_at = 0.0

    def __enter__(self):
        if not self._open_capture():
            self.opened = False
            return self
        for _ in range(50):
            frame = self._read_latest_once()
            if frame is not None:
                self._last_frame = frame
                self._last_frame_at = time.monotonic()
                self.opened = True
                print(f"VIDEO_INGEST\nbackend=opencv\nurl={self.source}\nresolution={self.width}x{self.height}")
                return self
            time.sleep(0.1)
        self._release_capture()
        self.opened = False
        return self

    def _open_capture(self):
        self._release_capture()
        lower = self.source.lower()
        if lower.startswith(("rtmp://", "rtmps://")):
            os.environ[OPENCV_STREAM_ENV] = os.getenv(OPENCV_STREAM_ENV, DEFAULT_RTMP_CAPTURE_OPTIONS)
        self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if self.capture is None or not self.capture.isOpened():
            return False
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        return True

    def _release_capture(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __exit__(self, *_):
        self._release_capture()

    def _resize(self, frame):
        if frame is None:
            return None
        height, width = frame.shape[:2]
        if width == self.width and height == self.height:
            return frame.copy()
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def _read_latest_once(self):
        if self.capture is None or not self.capture.isOpened():
            return None
        latest = None
        drain = max(1, OPENCV_FRAME_DRAIN)
        for _ in range(drain):
            ok, frame = self.capture.read()
            if ok and frame is not None:
                latest = frame
            elif latest is not None:
                break
            else:
                return None
        return self._resize(latest)

    def read(self):
        frame = self._read_latest_once()
        if frame is not None:
            if self._reconnecting:
                print("STREAM connected")
                self._reconnecting = False
                self._reconnect_attempt = 0
            self._miss_count = 0
            self._last_frame = frame
            self._last_frame_at = time.monotonic()
            return frame

        self._miss_count += 1
        now = time.monotonic()
        if self._miss_count >= 15 and now >= self._next_reconnect_at:
            if not self._reconnecting:
                print("STREAM disconnected reason=publisher_stopped")
                try:
                    from .speech_out import cancel_reply_capture

                    cancel_reply_capture("stream_disconnect")
                except Exception:
                    pass
                self._reconnecting = True
            self._reconnect_attempt += 1
            delay = min(8, 2 ** min(self._reconnect_attempt - 1, 3))
            print(f"STREAM reconnecting attempt={self._reconnect_attempt} nextRetrySeconds={delay}")
            self._next_reconnect_at = now + delay
            app_log.debug("stream ingest: reopening OpenCV capture")
            self._open_capture()
        stale = self._stale_frame()
        if stale is not None:
            return stale
        return None

    def _stale_frame(self):
        if self._last_frame is None:
            return None
        if time.monotonic() - self._last_frame_at > OPENCV_STREAM_STALE_SECONDS:
            return None
        return self._last_frame.copy()

    def show(self, frame):
        cv2.imshow("Gaze Assistant", frame)
        return cv2.waitKey(1) & 0xFF == ord("q")

    def seed_frame(self):
        return None if self._last_frame is None else self._last_frame.copy()


class _StreamCamera:
    def __init__(
        self,
        source,
        width=DEFAULT_STREAM_WIDTH,
        height=DEFAULT_STREAM_HEIGHT,
        target_fps=10,
        fallback_source=None,
    ):
        self.source = source
        self.fallback_source = fallback_source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.process = None
        self.opened = False
        self._frame_size = self.width * self.height * 3
        self._last_frame = None
        self._last_frame_at = 0.0
        self._restart_count = 0
        self._next_restart_at = 0.0
        self._reported_reconnected = False
        self._reported_disconnected = False
        self._apply_source_mode(source)

    def _apply_source_mode(self, source):
        self._is_hls = _is_hls_source(source)
        self._max_stale_seconds = 6.0 if self._is_hls else MAX_STALE_FRAME_SECONDS

    def __enter__(self):
        if self._wait_for_first_frame():
            return self
        if self.fallback_source and self.fallback_source != self.source:
            print(f"stream ingest: falling back to {self.fallback_source}")
            self._stop_process()
            self.source = self.fallback_source
            self._apply_source_mode(self.source)
            self._restart_count = 0
            self._next_restart_at = 0.0
            if self._wait_for_first_frame():
                return self
        self.opened = False
        return self

    def _wait_for_first_frame(self):
        self._start_process()
        for _ in range(40):
            frame = self._read_frame_once()
            if frame is not None:
                self._last_frame = frame
                self._last_frame_at = time.monotonic()
                self.opened = True
                return True
            if self.process is not None and self.process.poll() is not None:
                self._start_process()
            time.sleep(0.25)
        self.opened = False
        return False

    def __exit__(self, *_):
        self._stop_process()

    def _ffmpeg_cmd(self):
        scale = f"scale={self.width}:{self.height},setsar=1"
        if self._is_hls:
            # HLS segments arrive in bursts; avoid fps filter dropping the whole pipe.
            video_filter = scale
        else:
            video_filter = scale
        return [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            *_ffmpeg_input_args(self.source),
            "-fflags",
            "+nobuffer+genpts+discardcorrupt",
            "-flags",
            "low_delay",
            "-frame_drop_threshold",
            "0",
            "-thread_queue_size",
            "8",
            "-probesize",
            "500000",
            "-analyzeduration",
            "1000000",
            "-i",
            self.source,
            "-an",
            "-sn",
            "-vf",
            video_filter,
            "-r",
            str(self.target_fps),
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]

    def _start_process(self):
        self._stop_process()
        self.process = subprocess.Popen(
            self._ffmpeg_cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.2)
        if self.process.poll() is not None:
            err = ""
            try:
                err = (self.process.stderr.read() or b"").decode("utf-8", errors="replace").strip()
            except Exception:
                pass
            self.process = None
            self.opened = False
            if err:
                app_log.debug(f"stream ingest ffmpeg failed: {err[:500]}")
            return
        self.opened = self.process.stdout is not None
        self._restart_count += 1
        if self._restart_count > 1:
            delay = min(8, 2 ** min(self._restart_count - 2, 3))
            print(f"STREAM reconnecting attempt={self._restart_count - 1} nextRetrySeconds={delay}")
            app_log.debug(f"stream ingest: restarted ffmpeg ({self._restart_count - 1} reconnects)")
            self._reported_reconnected = False
        else:
            print(f"VIDEO_INGEST\nbackend=ffmpeg\nurl={self.source}\nresolution={self.width}x{self.height}\nfps={self.target_fps}")

    def _stop_process(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()
        self.process = None

    def _read_exact(self, size):
        if self.process is None or self.process.stdout is None:
            return None
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _stdout_available_bytes(self):
        if self.process is None or self.process.stdout is None:
            return 0
        available = array.array("i", [0])
        fcntl.ioctl(self.process.stdout.fileno(), termios.FIONREAD, available, True)
        return available[0]

    def _newest_available_raw_frame(self, raw):
        while self._stdout_available_bytes() >= self._frame_size:
            newer = self._read_exact(self._frame_size)
            if not newer:
                break
            raw = newer
        return raw

    def _restart_if_needed(self):
        if self.process is not None and self.process.poll() is None:
            return True
        if self._restart_count >= 30:
            return False
        now = time.monotonic()
        if now < self._next_restart_at:
            time.sleep(0.05)
            return self.process is not None and self.process.poll() is None
        delay = min(8.0, 2 ** min(max(self._restart_count - 1, 0), 3))
        self._next_restart_at = now + delay
        time.sleep(delay)
        self._start_process()
        return self.opened

    def _read_frame_once(self):
        raw = self._read_exact(self._frame_size)
        if not raw:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()

    def read(self):
        raw = self._read_exact(self._frame_size)
        if not raw:
            if not self._reported_disconnected:
                print("STREAM disconnected reason=publisher_stopped")
                try:
                    from .speech_out import cancel_reply_capture

                    cancel_reply_capture("stream_disconnect")
                except Exception:
                    pass
                self._reported_disconnected = True
        if not raw and not self._restart_if_needed():
            return self._stale_frame()
        if not raw:
            raw = self._read_exact(self._frame_size)
        if not raw:
            return self._stale_frame()
        raw = self._newest_available_raw_frame(raw)

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        self._last_frame = frame
        self._last_frame_at = time.monotonic()
        if self._restart_count > 1 and not self._reported_reconnected:
            print("STREAM connected")
            self._reported_reconnected = True
            self._reported_disconnected = False
        return frame

    def _stale_frame(self):
        if self._last_frame is None:
            return None
        if time.monotonic() - self._last_frame_at > self._max_stale_seconds:
            return None
        return self._last_frame.copy()

    def show(self, frame):
        cv2.imshow("Gaze Assistant", frame)
        return cv2.waitKey(1) & 0xFF == ord("q")

    def seed_frame(self):
        return None if self._last_frame is None else self._last_frame.copy()


class Camera:
    def __init__(self, source, fallback_source=None):
        self.source = source
        self.fallback_source = fallback_source
        self._impl = None
        self.opened = False
        self._lock = threading.RLock()
        self._last_good_frame = None
        self._last_good_frame_at = 0.0
        self._is_stream = _is_stream_source(source)

    def __enter__(self):
        if _is_stream_source(self.source):
            prefer_opencv = os.getenv("MENTRA_STREAM_BACKEND", "opencv").strip().lower() != "ffmpeg"
            use_opencv = prefer_opencv and self.source.lower().startswith(("rtmp://", "rtmps://", "http://", "https://"))
            if use_opencv:
                self._impl = _OpenCvStreamCamera(self.source)
                self._impl.__enter__()
                if not self._impl.opened:
                    print("stream ingest: OpenCV failed, trying ffmpeg pipe")
                    self._impl.__exit__(None, None, None)
                    self._impl = _StreamCamera(self.source, fallback_source=self.fallback_source)
                    self._impl.__enter__()
            else:
                self._impl = _StreamCamera(self.source, fallback_source=self.fallback_source)
                self._impl.__enter__()
        else:
            self._impl = _LocalCamera(self.source)
            self._impl.__enter__()
            print(f"VIDEO_INGEST\nbackend=opencv-local\nindex={self.source}")
        self.opened = bool(getattr(self._impl, "opened", False))
        if self.opened and hasattr(self._impl, "seed_frame"):
            seed = self._impl.seed_frame()
            if seed is not None:
                self._last_good_frame = seed
                self._last_good_frame_at = time.monotonic()
        if self.opened:
            cv2.namedWindow("Gaze Assistant", cv2.WINDOW_NORMAL)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._impl is not None:
            self._impl.__exit__(exc_type, exc, tb)
        cv2.destroyAllWindows()

    def read(self):
        with self._lock:
            if self._impl is None:
                return None
            frame = self._impl.read()
            if frame is not None:
                self._last_good_frame = frame
                self._last_good_frame_at = time.monotonic()
                return frame
            if self._last_good_frame is not None:
                hold = STREAM_GLITCH_HOLD_SECONDS if self._is_stream else MAX_STALE_FRAME_SECONDS
                if time.monotonic() - self._last_good_frame_at <= hold:
                    return self._last_good_frame.copy()
            return None

    def show(self, frame):
        with self._lock:
            if self._impl is None:
                return True
            return self._impl.show(frame)
