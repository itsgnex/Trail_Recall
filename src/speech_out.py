import threading
import shutil
import subprocess


_speaking = threading.Event()


def is_speaking():
    return _speaking.is_set()


def speak(text):
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
