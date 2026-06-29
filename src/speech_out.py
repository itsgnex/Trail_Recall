import shutil
import subprocess


def speak(text):
    print(f"assistant: {text}")
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
