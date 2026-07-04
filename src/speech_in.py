import tempfile
import time
import threading
import re

_whisper_model = None
_whisper_name = None
_whisper_warned = False
_audio_lock = threading.Lock()


def _normalize_text(text):
    text = (text or "").lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _wake_phrase_patterns(config):
    patterns = [
        r"^(hey|hay)\s+trail\s+recall\b",
        r"^(hey|hay)\s+trails?\b",
        r"^(hey|hay)\s+trial\b",
        r"^okay\s+trails?\b",
        r"^trail\s+recall\b",
        r"^(hey|hay)\s+assistant\b",
        r"^(hey|hay)\s+glasses\b",
    ]
    if getattr(config, "allow_single_word_wake", False):
        patterns.append(r"^trail\b")
    return patterns


def strip_wake_phrase(transcript, wake_phrases=None, allow_single_word_wake=False):
    text = _normalize_text(transcript)
    if not text:
        return ""
    class _Tmp:
        pass

    cfg = _Tmp()
    cfg.wake_phrases = tuple(wake_phrases or ())
    cfg.allow_single_word_wake = allow_single_word_wake
    for pattern in _wake_phrase_patterns(cfg):
        match = re.match(pattern, text)
        if match:
            remainder = text[match.end() :].strip()
            return remainder
    return text


def _wake_rejection_reason(transcript, config):
    text = _normalize_text(transcript)
    if not text:
        return "empty"

    word_count = len(text.split())
    if word_count < int(getattr(config, "wake_min_transcript_length", 2)):
        return "too short"

    if re.match(r"^(a|the)\s+trail\b", text):
        return "starts with article, likely normal speech"
    if re.match(r"^(hair|her)\s+trail\b", text):
        return "likely false wake variant"
    if re.match(r"^trail\b", text) and not getattr(config, "allow_single_word_wake", False):
        return "single-word wake disabled"
    if re.search(r"\btrail\b|\btrails\b|\btrial\b", text):
        return "contains trail-like word but not a wake phrase"
    return "not detected"


def wake_phrase_detected(transcript, config):
    text = _normalize_text(transcript)
    if not text:
        return False
    return any(re.match(pattern, text) for pattern in _wake_phrase_patterns(config))


def is_incomplete_command(command):
    text = _normalize_text(command)
    if not text:
        return True
    if text in {"can you hear me", "can u hear me", "are you there", "hello", "hey", "hey trail can you hear me"}:
        return True
    if text in {"can you", "can you tell me", "can you tell me what the", "what is the", "what is", "what are the", "describe", "read", "tell me"}:
        return True
    return bool(re.search(r"\b(can you|what is the|what are the|describe the|read the|tell me)\b\s*$", text))


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
            segments, _ = _whisper_model.transcribe(
                wav.name,
                language="en",
                vad_filter=bool(getattr(config, "whisper_use_vad", False)),
                initial_prompt=getattr(config, "whisper_initial_prompt", None),
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:
        if not _whisper_warned:
            print(f"Whisper STT unavailable: {exc}. Falling back to SpeechRecognition.")
            _whisper_warned = True
        return ""


def _record_audio(config, phrase_time_limit, typed_fallback=True, quiet=False, label="recording", mic_index=None):
    with _audio_lock:
        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()
            device_index = getattr(config, "mic_device_index", 0) if mic_index is None else mic_index
            names = sr.Microphone.list_microphone_names()
            if 0 <= device_index < len(names) and not quiet:
                print(f"using microphone: {names[device_index]}")
            with sr.Microphone(device_index=device_index) as source:
                timeout = phrase_time_limit if not typed_fallback else getattr(config, "mic_listen_timeout", 12)
                if not quiet:
                    print(f"{label} for {phrase_time_limit} seconds...")
                recognizer.adjust_for_ambient_noise(source, duration=getattr(config, "mic_ambient_noise_duration", 1))
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            return audio
        except Exception as exc:
            if exc.__class__.__name__ == "WaitTimeoutError" and not typed_fallback:
                return ""
            print(f"Microphone speech input unavailable: {exc}")
            if not typed_fallback:
                return ""
            print("Type your response instead, or allow microphone permission on macOS and install PyAudio.")
            try:
                return input("> ")
            except EOFError:
                return ""


def listen(config=None, typed_fallback=True, record_seconds=None, label="recording"):
    short_mode = bool(getattr(config, "whisper_short_reply_mode", True))
    record_seconds = int(record_seconds if record_seconds is not None else getattr(config, "whisper_record_seconds", 5))

    audio = _record_audio(config, record_seconds, typed_fallback, quiet=False, label=label)
    if isinstance(audio, str):
        return audio

    return _transcribe_with_fallback(audio, config, short_mode, typed_fallback, label)


def _transcribe_with_fallback(audio, config, short_mode, typed_fallback, label):
    text = transcribe_with_whisper(audio, config)
    if text:
        return text

    if short_mode:
        print("short reply retry...")
        audio = _record_audio(config, 3, typed_fallback, quiet=False, label=label)
        if isinstance(audio, str):
            return audio
        text = transcribe_with_whisper(audio, config)
        if text:
            return text

    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        return recognizer.recognize_google(audio)
    except Exception as exc:
        print(f"Microphone speech input unavailable: {exc}")
        if not typed_fallback:
            return ""
        print("Type your response instead, or allow microphone permission on macOS and install PyAudio.")
        try:
            return input("> ")
        except EOFError:
            return ""


def listen_for_wake_phrase(mic_index=None, config=None, state=None):
    if config is None:
        return None
    if not getattr(config, "voice_activation_mode", True) and not getattr(config, "wake_mode", False):
        return None
    print("wake listening...")
    audio = _record_audio(
        config,
        float(getattr(config, "wake_listen_seconds", 2.5)),
        typed_fallback=False,
        quiet=True,
        label="wake recording",
        mic_index=mic_index,
    )
    if isinstance(audio, str):
        return None

    transcript = transcribe_with_whisper(audio, config)
    debug = bool(getattr(config, "wake_debug_transcripts", True))
    log_empty = bool(getattr(config, "wake_log_empty_transcripts", False))
    normalized = _normalize_text(transcript)
    if not normalized:
        if debug and log_empty:
            print('wake heard: ""')
            print("wake phrase not detected")
        return None

    if debug or normalized:
        print(f'wake heard: "{normalized}"')

    if wake_phrase_detected(normalized, config):
        print(f'wake phrase detected: "{normalized}"')
        return transcript

    if getattr(state, "follow_up_enabled", False) or getattr(state, "awaiting_follow_up_reply", False):
        print(f'follow-up speech captured: "{normalized}"')
        return transcript

    reason = _wake_rejection_reason(normalized, config)
    if reason != "not detected":
        print(f"wake rejected: {reason}")
    else:
        print("wake phrase not detected")
    return None


def _demo():
    class C:
        wake_phrases = ("hey trail", "okay trail", "trail recall", "hey assistant", "hey glasses")
        allow_single_word_wake = False
        wake_debug_transcripts = True
        wake_log_empty_transcripts = False
        wake_min_transcript_length = 2

    assert strip_wake_phrase("Hey Trail, what is this?", C.wake_phrases, False) == "what is this"
    assert strip_wake_phrase("hey trails can you tell me what this is", C.wake_phrases, False) == "can you tell me what this is"
    assert strip_wake_phrase("hey trail", C.wake_phrases, False) == ""
    assert not wake_phrase_detected("a trail", C())
    assert wake_phrase_detected("hey trails can you tell me what this is", C())
    assert wake_phrase_detected("hey trial what is this", C())
    assert wake_phrase_detected("hay trail what is this", C())
    assert wake_phrase_detected("hey trail", C())
    assert wake_phrase_detected("hey trail recall what is this", C())
    assert not wake_phrase_detected("hair trail", C())
    assert not wake_phrase_detected("trail", C())
    assert is_incomplete_command("can you")
    assert is_incomplete_command("what is the")


if __name__ == "__main__":
    _demo()
