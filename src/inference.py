from __future__ import annotations

import contextlib
import threading

_lock = threading.Lock()
_wake_transcribing = threading.Event()


@contextlib.contextmanager
def local_model(priority: str = "normal"):
    if priority == "wake":
        _wake_transcribing.set()
    _lock.acquire()
    try:
        yield
    finally:
        _lock.release()
        if priority == "wake":
            _wake_transcribing.clear()


def visual_inference_allowed() -> bool:
    return not _wake_transcribing.is_set()
