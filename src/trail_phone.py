from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .config import DEFAULT_ANDROID_TRAIL_URL


def _trail_base(config) -> str:
    return (getattr(config, "android_trail_base_url", "") or "").strip().rstrip("/")


def _trail_timeout() -> float:
    return float(os.getenv("ANDROID_TRAIL_TIMEOUT", "12"))


def _trail_retries() -> int:
    return max(1, int(os.getenv("ANDROID_TRAIL_RETRIES", "3")))


def android_bridge_diagnostics(config) -> tuple[bool, str]:
    base = _trail_base(config)
    if not base:
        message = (
            "ANDROID_BRIDGE\n"
            "url=\n"
            "healthEndpoint=\n"
            "statusEndpoint=\n"
            "Android commands unavailable: ANDROID_TRAIL_URL is not set."
        )
        print(message)
        return False, "ANDROID_TRAIL_URL not set"

    health_endpoint = f"{base}/health"
    status_endpoint = f"{base}/trail/status"
    retries = _trail_retries()
    print(
        "ANDROID_BRIDGE\n"
        f"url={base}\n"
        f"healthEndpoint={health_endpoint}\n"
        f"statusEndpoint={status_endpoint}"
    )
    ok, detail = False, "not checked"
    for attempt in range(1, retries + 1):
        ok, detail = check_trail_health(config)
        if ok:
            break
        if attempt < retries:
            time.sleep(0.5 * attempt)
    _, status_detail = check_trail_status(config) if ok else (False, "not checked")
    if ok:
        print(f"ANDROID_BRIDGE\nstatus=CONNECTED\nurl={base}\nhealth={detail}\ntrailStatus={status_detail}")
        print("Android trail bridge connected.")
    else:
        print(
            "ANDROID_BRIDGE\n"
            "status=UNAVAILABLE\n"
            f"url={base}\n"
            f"error={detail}\n"
            f"retryCount={retries}\n"
            "Confirm the phone hotspot is connected and the Android navigation server is running."
        )
    return ok, detail


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


def check_trail_status(config) -> tuple[bool, str]:
    base = _trail_base(config)
    if not base:
        return False, "ANDROID_TRAIL_URL not set"
    url = f"{base}/trail/status"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=_trail_timeout()) as response:
            body = response.read().decode("utf-8", errors="replace")
            return 200 <= response.status < 300, body or f"HTTP {response.status}"
    except Exception as exc:
        return False, str(exc)


def send_trail_command(action: str, config) -> tuple[bool, str]:
    base = _trail_base(config)
    if not base:
        return False, f"Set ANDROID_TRAIL_URL to your phone, for example {DEFAULT_ANDROID_TRAIL_URL}"

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
