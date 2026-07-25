import threading
import time

from . import app_log


INTERACTION_PAUSE_TIMEOUT_SECONDS = 20.0
_lock = threading.Lock()
_reason = ""
_expires_at = 0.0


def pause_interaction(reason="android_decision_point", timeout_seconds=INTERACTION_PAUSE_TIMEOUT_SECONDS):
    global _reason, _expires_at
    with _lock:
        _reason = reason or "android_decision_point"
        _expires_at = time.monotonic() + max(0.01, timeout_seconds)
    print(f"INTERACTION_PAUSED reason={_reason}")


def resume_interaction(reason="android_decision_point"):
    global _reason, _expires_at
    with _lock:
        active_reason = _reason
        _reason = ""
        _expires_at = 0.0
    if active_reason:
        print(f"INTERACTION_RESUMED reason={active_reason}")


def interaction_is_paused():
    global _reason, _expires_at
    with _lock:
        if not _reason:
            return False
        if time.monotonic() < _expires_at:
            return True
        expired_reason = _reason
        _reason = ""
        _expires_at = 0.0
    print(f"INTERACTION_RESUMED reason={expired_reason} timeout=true")
    return False


def log_interaction_skip(event):
    app_log.rate_limited(f"interaction_paused_{event}", 2, f"{event} reason=interaction_paused", level="DEBUG")


def reset_interaction_pause_for_tests():
    global _reason, _expires_at
    with _lock:
        _reason = ""
        _expires_at = 0.0
