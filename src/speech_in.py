import tempfile

_whisper_model = None
_whisper_name = None
_whisper_warned = False


def transcribe_with_whisper(audio, config):
    global _whisper_model, _whisper_name, _whisper_warned
    if not getattr(config, "use_whisper_stt", True):
        return ""
    try:
        from faster_whisper import WhisperModel

        name = getattr(config, "whisper_model", "small.en")
        if _whisper_model is None or _whisper_name != name:
            print(f"loading whisper model: {name}")
            _whisper_model = WhisperModel(name, device="auto", compute_type="int8")
            _whisper_name = name
        else:
            print(f"using cached whisper model: {name}")

        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            wav.write(audio.get_wav_data())
            wav.flush()
            segments, _ = _whisper_model.transcribe(wav.name, language="en")
            return " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:
        if not _whisper_warned:
            print(f"Whisper STT unavailable: {exc}. Falling back to SpeechRecognition.")
            _whisper_warned = True
        return ""


def listen(config=None):
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        device_index = getattr(config, "mic_device_index", 0)
        names = sr.Microphone.list_microphone_names()
        if 0 <= device_index < len(names):
            print(f"using microphone: {names[device_index]}")
        with sr.Microphone(device_index=device_index) as source:
            timeout = getattr(config, "mic_listen_timeout", 12)
            print(f"listening for up to {timeout} seconds...")
            recognizer.adjust_for_ambient_noise(source, duration=getattr(config, "mic_ambient_noise_duration", 1))
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=getattr(config, "whisper_record_seconds", 7))
        text = transcribe_with_whisper(audio, config)
        return text or recognizer.recognize_google(audio)
    except Exception as exc:
        print(f"Microphone speech input unavailable: {exc}")
        print("Type your response instead, or allow microphone permission on macOS and install PyAudio.")
        try:
            return input("> ")
        except EOFError:
            return ""
