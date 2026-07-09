from __future__ import annotations

import json
import urllib.error
import urllib.request


def send_trail_command(action: str, config) -> tuple[bool, str]:
    base = (getattr(config, "android_trail_base_url", "") or "").strip().rstrip("/")
    if not base:
        return False, "Set ANDROID_TRAIL_URL to your phone, for example http://192.168.1.50:8766"

    url = f"{base}/trail/{action}"
    request = urllib.request.Request(url, method="POST", data=b"")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
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
        return False, str(exc)
