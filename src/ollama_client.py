import json
import socket
import time
import urllib.request

_warned = False


def generate(prompt, config, timeout=None, model=None, images=None):
    global _warned
    if not getattr(config, "use_ollama", True):
        return ""
    kind = "image" if images else "text"
    timeout = timeout or (config.ollama_image_timeout if images else config.ollama_text_timeout)
    model = model or getattr(config, "main_llm_model", config.ollama_model)
    start = time.monotonic()
    print(f"calling ollama model {model} for {kind}...")
    try:
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            **({"images": images} if images else {}),
        }).encode()
        request = urllib.request.Request(config.ollama_url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed = time.monotonic() - start
            print(f"ollama response received in {elapsed:.1f} seconds")
            return json.loads(response.read().decode()).get("response", "").strip()
    except (TimeoutError, socket.timeout) as exc:
        print(f"ollama timed out after {timeout} seconds")
        return ""
    except Exception as exc:
        if not _warned:
            print(f"Ollama unavailable: {exc}. Using local fallbacks.")
            _warned = True
        return ""


def generate_json(prompt, config, timeout=3, model=None, images=None):
    text = generate(prompt, config, timeout, model, images)
    if not text:
        return None
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None
