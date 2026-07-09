import threading
import shutil
import subprocess


_speaking = threading.Event()
_last_spoken_text = ""


def is_speaking():
    return _speaking.is_set()


def last_spoken_text():
    return _last_spoken_text


def speak(text):
    global _last_spoken_text
    _last_spoken_text = (text or "").strip()
    print(f"assistant: {text}")
    _speaking.set()
    try:
        if shutil.which("say"):
            try:
                subprocess.run(["say", text], check=False)
                return
            except Exception:
                pass
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            print(f"TTS unavailable: {exc}")
    finally:
        _speaking.clear()
