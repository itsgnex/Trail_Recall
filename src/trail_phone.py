from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


def _trail_base(config) -> str:
    return (getattr(config, "android_trail_base_url", "") or "").strip().rstrip("/")


def _trail_timeout() -> float:
    return float(os.getenv("ANDROID_TRAIL_TIMEOUT", "12"))


def _trail_retries() -> int:
    return max(1, int(os.getenv("ANDROID_TRAIL_RETRIES", "3")))


def format_trail_error(detail: str) -> str:
    text = (detail or "").strip()
    lower = text.lower()
    if "set android_trail_url" in lower:
        return text
    if "timed out" in lower or "timeout" in lower:
        return (
            "I couldn't reach your phone. Keep Trail Return Lab open, "
            "tap Glasses → Start everything, and check ANDROID_TRAIL_URL matches your phone's Wi‑Fi IP."
        )
    if "connection refused" in lower or "no route to host" in lower or "network is unreachable" in lower:
        return (
            "The phone trail server isn't reachable. Same Wi‑Fi as the Mac, "
            "app open with Glasses → Start everything, then verify the IP in ANDROID_TRAIL_URL."
        )
    return "Sorry, I couldn't reach the trail app on your phone."


def check_trail_health(config) -> tuple[bool, str]:
    base = _trail_base(config)
    if not base:
        return False, "ANDROID_TRAIL_URL not set"
    url = f"{base}/health"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=_trail_timeout()) as response:
            body = response.read().decode("utf-8", errors="replace")
            if response.status == 200:
                return True, body or "ok"
            return False, body or f"HTTP {response.status}"
    except Exception as exc:
        return False, str(exc)


def send_trail_command(action: str, config) -> tuple[bool, str]:
    base = _trail_base(config)
    if not base:
        return False, "Set ANDROID_TRAIL_URL to your phone, for example http://192.168.1.50:8766"

    url = f"{base}/trail/{action}"
    timeout = _trail_timeout()
    retries = _trail_retries()
    last_error = "unknown error"

    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, method="POST", data=b"")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    return response.status == 200, body or "ok"
                if payload.get("ok"):
                    return True, payload.get("action") or action
                return False, payload.get("error") or body
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return False, detail or f"HTTP {exc.code}"
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                print(f"trail phone: {action} attempt {attempt}/{retries} failed ({last_error}), retrying...")
                time.sleep(0.6 * attempt)
                continue

    from .audio_http import enqueue_trail_command

    enqueue_trail_command(action)
    print(f"trail phone: {action} queued on Mac :8765/pending-trail (phone will poll while streaming)")
    return True, action
