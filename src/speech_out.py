import os
import shutil
import subprocess
import threading
import urllib.parse
import urllib.request
import uuid

from .audio_http import enqueue_glasses_speech


_speaking = threading.Event()
_last_spoken_text = ""


def _glasses_audio_enabled():
    return bool((os.getenv("ANDROID_TRAIL_URL", "") or "").strip())


def _mirror_to_android_glasses(text):
    base = (os.getenv("ANDROID_TRAIL_URL", "") or "").strip().rstrip("/")
    if not base or not text:
        return False
    try:
        query = urllib.parse.urlencode({"text": text})
        request = urllib.request.Request(
            f"{base}/trail/speak?{query}",
            method="POST",
            data=b"",
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            ok = 200 <= response.status < 300
            if ok:
                print("glasses audio: sent to phone trail server for Bluetooth playback")
            return ok
    except Exception as exc:
        print(f"glasses audio: phone trail server unavailable ({exc})")
        return False


def is_speaking():
    return _speaking.is_set()


def last_spoken_text():
    return _last_spoken_text


def speak(text):
    global _last_spoken_text
    _last_spoken_text = (text or "").strip()
    event_id = uuid.uuid4().hex[:8]
    print(f"assistant: {text}")
    _speaking.set()
    try:
        if _glasses_audio_enabled():
            if _mirror_to_android_glasses(_last_spoken_text):
                print(f'SPEECH_OUT\ntarget=ANDROID\neventId={event_id}\ntext="{_last_spoken_text}"')
                return
            enqueue_glasses_speech(_last_spoken_text)
            print(
                "glasses audio: queued on Mac :8765/pending — phone picks this up while streaming "
                "(Glasses -> Start everything)"
            )
            print(f'SPEECH_OUT\ntarget=ANDROID_QUEUE\neventId={event_id}\ntext="{_last_spoken_text}"')
            return
        if shutil.which("say"):
            try:
                print(f'SPEECH_OUT\ntarget=MAC_LOCAL\nreason=ANDROID_UNAVAILABLE\neventId={event_id}')
                subprocess.run(["say", text], check=False)
                return
            except Exception:
                pass
        try:
            import pyttsx3

            print(f'SPEECH_OUT\ntarget=MAC_LOCAL\nreason=ANDROID_UNAVAILABLE\neventId={event_id}')
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            print(f"TTS unavailable: {exc}")
    finally:
        _speaking.clear()
