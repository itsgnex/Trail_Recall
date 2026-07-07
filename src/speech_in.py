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
    phrases = list(dict.fromkeys(_normalize_text(phrase) for phrase in getattr(config, "wake_phrases", ()) if _normalize_text(phrase)))
    if not phrases:
        phrases = [
            "hey trail",
            "okay trail",
            "trail recall",
            "hey assistant",
            "hey glasses",
            "hey nova",
            "nova",
        ]

    patterns = []
    for phrase in sorted(phrases, key=lambda value: (-len(value.split()), value)):
        if phrase == "hey trail":
            patterns.append(r"^(hey|hay)\s+(trail|trails|trial|drell|drill|girl|twelve|12)\b")
            continue
        if phrase == "okay trail":
            patterns.append(r"^(okay|ok)\s+(trail|trails|trial|drell|drill|girl|twelve|12)\b")
            continue
        if phrase == "trail recall":
            patterns.append(r"^(trail|trails|trial)\s+recall\b")
            continue
        words = phrase.split()
        escaped = r"\s+".join(re.escape(word) for word in words)
        if len(words) > 1:
            patterns.append(rf"^{escaped}\b")
            continue
        if phrase == "nova":
            patterns.append(rf"^{escaped}\b")
        elif phrase == "trail":
            if getattr(config, "allow_single_word_wake", False):
                patterns.append(rf"^{escaped}\b")
            else:
                patterns.append(rf"^{escaped}\b\s+.+")
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
    if text == "trail" and not getattr(config, "allow_single_word_wake", False):
        return "single-word wake disabled"
    if re.search(r"\btrail\b|\btrails\b|\btrial\b", text):
        return "contains trail-like word but not a wake phrase"
    return "not detected"


def wake_phrase_detected(transcript, config):
    text = _normalize_text(transcript)
    if not text:
        return False
    if any(re.match(pattern, text) for pattern in _wake_phrase_patterns(config)):
        return True
    if re.search(r"\b(can you|could you|would you)\s+(hear|help)\s+(me|that|this)?\b", text):
        return True
    if re.match(r"^(hey|hay|okay|ok|yo)\b", text) and re.search(
        r"\b(can you|could you|would you|do it|help me|hear me|what is|what's|read this|look at this|tell me|please do)\b",
        text,
    ):
        return True
    return False


def is_wake_check_command(command):
    text = _normalize_text(command)
    return bool(re.search(r"\b(can you|could you|would you)\s+hear\s+(me|that|this)?\b|\bare you there\b|\bhello\b", text))


def is_incomplete_command(command):
    text = _normalize_text(command)
    if not text:
        return True
    if text in {"hey"}:
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


def transcribe_with_openai(audio, config):
    api_key = getattr(config, "openai_api_key", "")
    if not api_key:
        print("OpenAI STT skipped: OPENAI_API_KEY is not set.")
        return ""
    try:
        import requests

        model = getattr(config, "openai_stt_model", "gpt-4o-mini-transcribe")
        url = getattr(config, "openai_stt_url", "https://api.openai.com/v1/audio/transcriptions")
        timeout = int(getattr(config, "openai_stt_timeout", 8))
        print(f"calling OpenAI STT model: {model}")
        started_at = time.monotonic()
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("audio.wav", audio.get_wav_data(), "audio/wav")},
            data={"model": model, "response_format": "text", "language": "en"},
            timeout=timeout,
        )
        if response.status_code >= 400:
            print(f"OpenAI STT failed: {response.status_code} {response.text[:180]}")
            return ""
        print(f"OpenAI STT response received in {time.monotonic() - started_at:.1f} seconds")
        return response.text.strip()
    except Exception as exc:
        print(f"OpenAI STT unavailable: {exc}")
        return ""


def _transcribe_primary(audio, config):
    provider = getattr(config, "stt_provider", "local")
    if provider == "openai":
        return transcribe_with_openai(audio, config)
    if provider == "openai_first":
        return transcribe_with_openai(audio, config) or transcribe_with_whisper(audio, config)
    return transcribe_with_whisper(audio, config)


def _transcribe_with_google(audio):
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    return recognizer.recognize_google(audio)


def _record_audio(config, phrase_time_limit, typed_fallback=True, quiet=False, label="recording", mic_index=None, fixed_duration=False):
    with _audio_lock:
        last_error = None
        for attempt in range(2):
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
                    if fixed_duration:
                        audio = recognizer.record(source, duration=phrase_time_limit)
                    else:
                        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                return audio
            except Exception as exc:
                last_error = exc
                if exc.__class__.__name__ == "WaitTimeoutError" and not typed_fallback:
                    return ""
                if attempt == 0:
                    print("Microphone had a brief issue; retrying once...")
                    time.sleep(0.25)
                    continue
                print(f"Microphone speech input unavailable: {last_error}")
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
    text = _transcribe_primary(audio, config)
    if text:
        return text

    if short_mode:
        print("short reply retry...")
        audio = _record_audio(config, 3, typed_fallback, quiet=False, label=label)
        if isinstance(audio, str):
            return audio
        text = _transcribe_primary(audio, config)
        if text:
            return text

    try:
        return _transcribe_with_google(audio)
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
        fixed_duration=True,
    )
    if isinstance(audio, str):
        return None

    transcript = _transcribe_primary(audio, config)
    normalized = _normalize_text(transcript)
    if not wake_phrase_detected(normalized, config):
        try:
            google_transcript = _transcribe_with_google(audio)
            if wake_phrase_detected(google_transcript, config):
                transcript = google_transcript
                normalized = _normalize_text(transcript)
            elif not normalized:
                transcript = google_transcript
                normalized = _normalize_text(transcript)
        except Exception as exc:
            if not _whisper_warned:
                print(f"Wake fallback STT unavailable: {exc}.")
            transcript = transcript or ""
            normalized = _normalize_text(transcript)
    debug = bool(getattr(config, "wake_debug_transcripts", True))
    log_empty = bool(getattr(config, "wake_log_empty_transcripts", False))
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
        wake_phrases = ("hey trail", "okay trail", "trail recall", "hey assistant", "hey glasses", "hey nova", "nova")
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
    assert wake_phrase_detected("hey nova what is this", C())
    assert wake_phrase_detected("nova, what is this", C())
    assert wake_phrase_detected("tell can you hear me", C())
    assert is_wake_check_command("can you hear me")
    assert not wake_phrase_detected("hair trail", C())
    assert not wake_phrase_detected("trail", C())
    assert is_incomplete_command("can you")
    assert is_incomplete_command("what is the")


if __name__ == "__main__":
    _demo()
