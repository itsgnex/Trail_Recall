from __future__ import annotations

import os
import time

LEVELS = {"ERROR": 40, "WARN": 30, "INFO": 20, "DEBUG": 10, "TRACE": 5}
_last = {}
_counts = {}


def level_name() -> str:
    return os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"


def enabled(level: str) -> bool:
    return LEVELS.get(level.upper(), 20) >= LEVELS.get(level_name(), 20)


def compact() -> bool:
    return os.getenv("TERMINAL_COMPACT_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}


def truncate(text: str, limit: int = 120) -> str:
    value = str(text or "").replace("\n", " ")
    return value if len(value) <= limit else value[: max(0, limit - 3)] + "..."


def log(level: str, message: str):
    if enabled(level):
        print(message)


def debug(message: str):
    log("DEBUG", message)


def info(message: str):
    log("INFO", message)


def warn(message: str):
    log("WARN", message)


def error(message: str):
    log("ERROR", message)


def rate_limited(key: str, seconds: float, message: str, level: str = "INFO"):
    now = time.monotonic()
    last = _last.get(key)
    if last is None or now - last >= seconds:
        _last[key] = now
        if _counts.get(key):
            message = f"{message} suppressed={_counts.pop(key)}"
        log(level, message)
        return True
    _counts[key] = _counts.get(key, 0) + 1
    return False
