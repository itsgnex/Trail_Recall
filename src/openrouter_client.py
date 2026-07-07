import json
import socket
import time
import urllib.request

_warned = False


def image_to_data_url(image_base64, mime_type="image/jpeg"):
    return f"data:{mime_type};base64,{image_base64}"


def generate(prompt, config, timeout=None, model=None, images=None):
    global _warned
    api_key = getattr(config, "openrouter_api_key", "")
    if not api_key:
        return ""

    timeout = timeout or getattr(config, "openrouter_timeout", 8)
    model = model or getattr(config, "openrouter_model", "google/gemini-3.1-flash-lite")
    url = getattr(config, "openrouter_url", "https://openrouter.ai/api/v1/chat/completions")
    start = time.monotonic()

    content = [{"type": "text", "text": prompt}]
    if images:
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Metra-Live",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed = time.monotonic() - start
            print(f"openrouter response received in {elapsed:.1f} seconds")
            payload = json.loads(response.read().decode())
            choices = payload.get("choices") or []
            if not choices:
                return ""
            message = (choices[0].get("message") or {})
            return (message.get("content") or "").strip()
    except (TimeoutError, socket.timeout):
        print(f"openrouter timed out after {timeout} seconds")
        return ""
    except Exception as exc:
        if not _warned:
            print(f"OpenRouter unavailable: {exc}. Using local fallbacks.")
            _warned = True
        return ""
