import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


STREAM_SCHEMES = ("http://", "https://", "rtmp://", "rtsp://", "udp://", "srt://")


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


class _StreamCamera:
    def __init__(self, source, width=640, height=360):
        self.source = source
        self.width = width
        self.height = height
        self.process = None
        self.opened = False
        self._frame_size = self.width * self.height * 3

    def __enter__(self):
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-i",
            self.source,
            "-an",
            "-sn",
            "-vf",
            f"scale={self.width}:{self.height},setsar=1",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.opened = self.process.poll() is None and self.process.stdout is not None
        return self

    def __exit__(self, *_):
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

    def read(self):
        raw = self._read_exact(self._frame_size)
        if not raw:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame.copy()

    def show(self, frame):
        cv2.imshow("Gaze Assistant", frame)
        return cv2.waitKey(1) & 0xFF == ord("q")


class Camera:
    def __init__(self, source):
        self.source = source
        self._impl = None
        self.opened = False

    def __enter__(self):
        if _is_stream_source(self.source):
            self._impl = _StreamCamera(self.source)
        else:
            self._impl = _LocalCamera(self.source)
        self._impl.__enter__()
        self.opened = bool(getattr(self._impl, "opened", False))
        if self.opened:
            cv2.namedWindow("Gaze Assistant", cv2.WINDOW_NORMAL)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._impl is not None:
            self._impl.__exit__(exc_type, exc, tb)
        cv2.destroyAllWindows()

    def read(self):
        if self._impl is None:
            return None
        return self._impl.read()

    def show(self, frame):
        if self._impl is None:
            return True
        return self._impl.show(frame)
