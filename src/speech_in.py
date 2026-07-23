import audioop
from collections import deque
import difflib
import multiprocessing as mp
import tempfile
import time
import threading
import re
import traceback
from . import app_log

_whisper_model = None
_whisper_name = None
_wake_whisper_model = None
_wake_whisper_name = None
_whisper_warned = False
_audio_lock = threading.Lock()
_mic_calibrated = False
_saved_energy_threshold = None
_wake_ambient_rms = 0
_wake_noise_counts = {"below_dynamic_threshold": 0, "too_short": 0}
_wake_noise_last = 0.0
GLASSES_PCM_SILENCE_RMS = 120
POST_TTS_LISTEN_DELAY_SECONDS = 0.5
COMMAND_END_SILENCE_SECONDS = 0.7
SHORT_REPLY_WORDS = {"yes", "yeah", "okay", "ok", "no", "stop", "continue", "left", "right"}
FILLER_WORDS = {"hey", "hay", "look", "okay", "ok", "um", "uh", "ah", "er"}


def _speech_end_silence_seconds(config):
    return max(0.05, float(getattr(config, "speech_end_silence_ms", 450)) / 1000.0)


def _silence_seconds(config, silence_ms=None):
    if silence_ms is None:
        return _speech_end_silence_seconds(config)
    return max(0.05, float(silence_ms) / 1000.0)


def _glasses_silence_rms(seconds, fixed_duration=False):
    if not fixed_duration and float(seconds) <= 2.0:
        return 60
    return GLASSES_PCM_SILENCE_RMS


def _wake_dynamic_threshold(config):
    return max(
        int(getattr(config, "wake_min_rms", 140)),
        int(_wake_ambient_rms * float(getattr(config, "wake_ambient_multiplier", 2.5))),
    )


def audio_signal_stats(audio, threshold=120):
    raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
    if len(raw) % 2:
        raw = raw[:-1]
    if not raw:
        return {"rms": 0, "peakRms": 0, "voicedMs": 0, "durationMs": 0}
    chunk_bytes = int(16000 * 2 * 0.05)
    rms_values = []
    voiced = 0
    for offset in range(0, len(raw), chunk_bytes):
        chunk = raw[offset : offset + chunk_bytes]
        if len(chunk) < 2:
            continue
        rms = audioop.rms(chunk, 2)
        rms_values.append(rms)
        if rms >= threshold:
            voiced += 1
    return {
        "rms": int(audioop.rms(raw, 2)),
        "peakRms": max(rms_values) if rms_values else 0,
        "voicedMs": int(voiced * 50),
        "durationMs": int(len(raw) / (16000 * 2) * 1000),
    }


def reply_audio_is_usable(audio, config):
    threshold = int(getattr(config, "reply_min_rms", 120))
    stats = audio_signal_stats(audio, threshold=threshold)
    ok = (
        stats["rms"] >= threshold
        and stats["peakRms"] >= threshold
        and stats["voicedMs"] >= int(getattr(config, "reply_min_voiced_ms", 150))
    )
    if not ok:
        print(f"REPLY_AUDIO_REJECTED reason=low_signal rms={stats['rms']} voicedMs={stats['voicedMs']}")
    return ok, stats


def _log_wake_audio_rejected(reason, rms, threshold):
    global _wake_noise_last
    _wake_noise_counts[reason] = _wake_noise_counts.get(reason, 0) + 1
    now = time.monotonic()
    if now - _wake_noise_last >= 10.0:
        rejected = sum(_wake_noise_counts.values())
        app_log.debug(
            "WAKE_NOISE_SUMMARY "
            f"rejected={rejected} "
            f"belowThreshold={_wake_noise_counts.get('below_dynamic_threshold', 0)} "
            f"tooShort={_wake_noise_counts.get('too_short', 0)} "
            "windowSeconds=10"
        )
        _wake_noise_counts.clear()
        _wake_noise_last = now


def calibrate_wake_ambient(config, seconds=0.5):
    if not getattr(config, "use_glasses_mic", False):
        return 0
    try:
        from .mic_ingest import glasses_mic_buffer

        chunk_bytes = int(16000 * 2 * 0.05)
        deadline = time.monotonic() + float(seconds)
        samples = []
        buffer = glasses_mic_buffer()
        while time.monotonic() < deadline:
            chunk = buffer.read_bytes(chunk_bytes, timeout=0.05)
            if chunk:
                if len(chunk) % 2:
                    chunk = chunk[:-1]
                rms = audioop.rms(chunk, 2)
                if rms > 0:
                    samples.append(rms)
        global _wake_ambient_rms
        if not samples:
            print(f"WAKE_AMBIENT_RMS skipped=zero_audio threshold={_wake_dynamic_threshold(config)}")
            return _wake_ambient_rms
        _wake_ambient_rms = int(sum(samples) / len(samples))
        print(f"WAKE_AMBIENT_RMS rms={_wake_ambient_rms} threshold={_wake_dynamic_threshold(config)}")
        return _wake_ambient_rms
    except Exception as exc:
        print(f"WAKE_AMBIENT_RMS unavailable: {exc}")
        return 0


def _elapsed_ms(started_at):
    return int((time.monotonic() - started_at) * 1000)


def _latency_stage(prefix, event):
    if prefix == "WAKE_AUDIO_CAPTURE":
        return {
            "start": "WAKE_LISTEN_START",
            "speech": "WAKE_SPEECH_STARTED",
            "silence": "WAKE_SILENCE_DETECTED",
            "end": "WAKE_AUDIO_CAPTURE_END",
        }[event]
    return {
        "start": f"{prefix}_START",
        "speech": f"{prefix.replace('CAPTURE', 'SPEECH')}_STARTED",
        "silence": f"{prefix.replace('CAPTURE', 'SILENCE')}_DETECTED",
        "end": f"{prefix}_END",
    }[event]


def _normalize_text(text):
    text = (text or "").lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _preview(text, limit=120):
    return app_log.truncate(_normalize_text(text), limit=limit)


def wake_transcript_rejection(text):
    words = _normalize_text(text).split()
    if not words:
        return None
    if len(words) < 6:
        return None
    unique = set(words)
    max_run = 1
    run = 1
    for prev, word in zip(words, words[1:]):
        run = run + 1 if word == prev else 1
        max_run = max(max_run, run)
    most_common = max(words.count(word) for word in unique)
    if len(words) > 20 or max_run > 4 or most_common / len(words) > 0.5 or len(unique) / len(words) < 0.35:
        return {
            "reason": "repetition_hallucination",
            "tokens": len(words),
            "unique": len(unique),
            "preview": " ".join(words[:4]) + ("..." if len(words) > 4 else ""),
        }
    return None


def _is_repeated_filler_command(text):
    words = _normalize_text(text).split()
    return bool(words) and set(words).issubset(FILLER_WORDS)


def _is_complete_short_reply(text):
    return _normalize_text(text) in SHORT_REPLY_WORDS


def _wake_phrase_patterns(config):
    phrases = list(dict.fromkeys(_normalize_text(phrase) for phrase in getattr(config, "wake_phrases", ()) if _normalize_text(phrase)))
    if not phrases:
        phrases = [
            "hey look",
            "okay look",
            "hey assistant",
            "hey glasses",
            "hey nova",
            "nova",
        ]

    patterns = []
    for phrase in sorted(phrases, key=lambda value: (-len(value.split()), value)):
        if phrase == "hey look":
            patterns.append(r"^(hey|hay)\s+(look|luck|lok)\b")
            continue
        if phrase == "okay look":
            patterns.append(r"^(okay|ok)\s+(look|luck|lok)\b")
            continue
        if phrase == "hey trail":
            patterns.append(r"^(hey|hay)\s+(trail|trails|trial|troll|trale|drell|drill|girl|twelve|12)\b")
            continue
        if phrase == "okay trail":
            patterns.append(r"^(okay|ok)\s+(trail|trails|trial|troll|trale|drell|drill|girl|twelve|12)\b")
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
            patterns.append(r"^(trail|trails|trial|troll|trale)\b")
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
    return False


def direct_voice_command_detected(transcript, config):
    """Allow image/command requests without a wake phrase when voice_command_mode is on."""
    if not getattr(config, "voice_command_mode", True):
        return False
    text = f" {_normalize_text(transcript)} "
    if len(text.strip()) < 4:
        return False
    return bool(
        re.search(
            r"\b("
            r"can you tell me what this is|can you tell me what that is|"
            r"what is this|what is that|what am i looking at|what's this|what's that|"
            r"describe this|describe it|describe what i am looking at|what do you see|"
            r"read this|read the sign|read that sign|read a sign|read this sign|"
            r"can you read a sign|can you read the sign|can you read this sign|can you read it|"
            r"what does this sign mean|what does that sign mean|"
            r"what plant is this|identify this plant|tell me about this plant|"
            r"start the trail|start trail|begin the trail|begin trail|record the trail|record trail|record my route|record route|"
            r"stop the trail|stop trail|end the trail|end trail|finish the trail|"
            r"take me back|navigate back|lead me back|guide me back|bring me back|"
            r"choose left|choose right|destination reached|i reached the destination|"
            r"help me|can you help me|look at this|tell me what this is"
            r")\b",
            text,
        )
    )


def is_wake_check_command(command):
    text = _normalize_text(command)
    return bool(re.search(r"\b(can you|could you|would you)\s+hear\s+(me|that|this)?\b|\bare you there\b|\bhello\b", text))


def is_incomplete_command(command):
    text = _normalize_text(command)
    if not text:
        return True
    if text in {"hey"}:
        return True
    if _is_repeated_filler_command(text):
        return True
    if text in {"can you", "can you tell me", "can you tell me what the", "what is the", "what is", "what are the", "describe", "read", "tell me"}:
        return True
    return bool(re.search(r"\b(can you|what is the|what are the|describe the|read the|tell me)\b\s*$", text))


def preload_whisper_model(config):
    """Load only the wake Whisper model at startup; fallback STT stays lazy."""
    if not getattr(config, "use_whisper_stt", True):
        return
    try:
        from faster_whisper import WhisperModel
        from .latency import log_stage

        global _wake_whisper_model, _wake_whisper_name
        name = getattr(config, "wake_whisper_model", "tiny.en")
        if _wake_whisper_model is None or _wake_whisper_name != name:
            log_stage("WAKE_MODEL_LOAD_START", model=name)
            print(f"preloading wake whisper model: {name}")
            _wake_whisper_model = WhisperModel(
                name,
                device="cpu",
                compute_type="int8",
                cpu_threads=int(getattr(config, "wake_whisper_cpu_threads", 4)),
                num_workers=1,
            )
            _wake_whisper_name = name
            log_stage("WAKE_MODEL_LOAD_END", model=name)
            print("wake whisper model ready")
    except Exception as exc:
        print(f"Whisper preload skipped: {exc}")


def transcribe_wake_with_whisper(audio, config):
    global _wake_whisper_model, _wake_whisper_name
    if not getattr(config, "use_whisper_stt", True):
        return ""
    try:
        from faster_whisper import WhisperModel
        from .inference import local_model as local_inference_model
        from .latency import log_stage

        name = getattr(config, "wake_whisper_model", "tiny.en")
        if _wake_whisper_model is None or _wake_whisper_name != name:
            log_stage("WAKE_MODEL_LOAD_START", model=name)
            print(f"loading wake whisper model: {name}")
            _wake_whisper_model = WhisperModel(
                name,
                device="cpu",
                compute_type="int8",
                cpu_threads=int(getattr(config, "wake_whisper_cpu_threads", 4)),
                num_workers=1,
            )
            _wake_whisper_name = name
            log_stage("WAKE_MODEL_LOAD_END", model=name)
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            wav.write(audio.get_wav_data())
            wav.flush()
            result = {"text": ""}

            def _run():
                with local_inference_model("wake"):
                    segments, _ = _wake_whisper_model.transcribe(
                        wav.name,
                        language="en",
                        vad_filter=False,
                        beam_size=1,
                        best_of=1,
                        temperature=0,
                        condition_on_previous_text=False,
                        initial_prompt="hey trail, okay trail, hey look, okay look, hey glasses, hey assistant",
                    )
                    parts = []
                    avg_log_probs = []
                    no_speech_probs = []
                    for segment in segments:
                        parts.append(segment.text.strip())
                        if hasattr(segment, "avg_logprob"):
                            avg_log_probs.append(float(segment.avg_logprob))
                        if hasattr(segment, "no_speech_prob"):
                            no_speech_probs.append(float(segment.no_speech_prob))
                    result["text"] = " ".join(parts).strip()
                    result["avgLogProb"] = sum(avg_log_probs) / len(avg_log_probs) if avg_log_probs else None
                    result["noSpeechProb"] = max(no_speech_probs) if no_speech_probs else None

            worker = threading.Thread(target=_run, name="WakeWhisper", daemon=True)
            worker.start()
            worker.join(float(getattr(config, "wake_transcription_timeout_seconds", 2.0)))
            if worker.is_alive():
                log_stage("WAKE_TRANSCRIPTION_TIMEOUT", timeoutSeconds=getattr(config, "wake_transcription_timeout_seconds", 2.0))
                return None
            if result.get("noSpeechProb") is not None and result["noSpeechProb"] > 0.8:
                return ""
            if result.get("avgLogProb") is not None and result["avgLogProb"] < -1.2:
                return ""
            return result["text"]
    except Exception as exc:
        _log_transcription_error("Wake Whisper STT unavailable", exc)
        return ""


def _wait_after_tts(config, after_tts=False):
    from .speech_out import is_speaking

    was_speaking = False
    while is_speaking():
        was_speaking = True
        time.sleep(0.05)
    if not (after_tts or was_speaking):
        return
    if getattr(config, "use_glasses_mic", False):
        from .mic_ingest import glasses_mic_buffer

        glasses_mic_buffer().clear()
    delay = max(POST_TTS_LISTEN_DELAY_SECONDS, float(getattr(config, "mic_post_tts_delay", 0.0) or 0.0))
    if delay > 0:
        time.sleep(delay)
    if getattr(config, "use_glasses_mic", False):
        from .mic_ingest import glasses_mic_buffer

        glasses_mic_buffer().clear()


def _prepare_microphone(recognizer, source, config):
    """Calibrate once per session; use a short refresh afterward so speech is not eaten."""
    global _mic_calibrated, _saved_energy_threshold

    full_duration = float(getattr(config, "mic_ambient_noise_duration", 1.0))
    refresh_duration = float(getattr(config, "mic_ambient_refresh_duration", 0.25))

    recognizer.dynamic_energy_threshold = True
    if _mic_calibrated and _saved_energy_threshold is not None:
        recognizer.energy_threshold = _saved_energy_threshold
        duration = refresh_duration
    else:
        duration = full_duration

    recognizer.adjust_for_ambient_noise(source, duration=duration)
    _saved_energy_threshold = recognizer.energy_threshold
    _mic_calibrated = True


def _whisper_process_worker(wav_data, model_name, vad_filter, initial_prompt, output):
    try:
        from faster_whisper import WhisperModel

        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            wav.write(wav_data)
            wav.flush()
            model = WhisperModel(model_name, device="auto", compute_type="int8")
            segments, _ = model.transcribe(
                wav.name,
                language="en",
                vad_filter=vad_filter,
                initial_prompt=initial_prompt,
            )
            output.put(" ".join(segment.text.strip() for segment in segments).strip())
    except Exception as exc:
        output.put({"error": repr(exc)})


def _transcribe_with_whisper_process(audio, config, timeout):
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_whisper_process_worker,
        args=(
            audio.get_wav_data(),
            getattr(config, "whisper_model", "small.en"),
            bool(getattr(config, "whisper_use_vad", False)),
            getattr(config, "whisper_initial_prompt", None),
            queue,
        ),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(0.5)
        if process.is_alive():
            process.kill()
            process.join(0.5)
        print(f"LOCAL_STT_FALLBACK_TIMEOUT timeoutSeconds={timeout}")
        return ""
    if queue.empty():
        return ""
    result = queue.get()
    if isinstance(result, dict):
        print(f"Whisper STT unavailable: {result.get('error')}")
        return ""
    return result


def transcribe_with_whisper(audio, config, force=False):
    global _whisper_model, _whisper_name, _whisper_warned
    if not force and not getattr(config, "use_whisper_stt", True):
        return ""
    try:
        from .latency import log_stage

        name = getattr(config, "whisper_model", "small.en")
        timeout = float(getattr(config, "local_stt_timeout_seconds", 2.0))
        log_stage("FALLBACK_MODEL_LOAD_START", model=name)
        with _audio_lock:
            text = _transcribe_with_whisper_process(audio, config, timeout)
        log_stage("FALLBACK_MODEL_LOAD_END", model=name, timedOut=not bool(text))
        return text
    except Exception as exc:
        _log_transcription_error("Whisper STT unavailable", exc)
        _whisper_warned = True
        return ""


def transcribe_with_openai(audio, config):
    api_key = getattr(config, "openai_api_key", "")
    if not api_key:
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


def transcribe_with_openrouter(audio, config):
    api_key = getattr(config, "openrouter_api_key", "")
    model = getattr(config, "openrouter_stt_model", "openai/gpt-4o-mini-transcribe")
    if not api_key:
        print(f"LOCAL_STT_FALLBACK provider=openrouter model={model} reason=missing_api_key")
        return ""
    try:
        import requests

        url = "https://openrouter.ai/api/v1/audio/transcriptions"
        timeout = float(getattr(config, "openrouter_stt_timeout_seconds", 3.0))
        wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
        started_at = time.monotonic()
        from .latency import log_stage

        log_stage("OPENROUTER_STT_START", provider="openrouter", model=model)
        app_log.debug(f"OPENROUTER_STT_START provider=openrouter model={model}")
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("audio.wav", wav_data, "audio/wav")},
            data={"model": model, "response_format": "json", "language": "en"},
            timeout=timeout,
        )
        log_stage("OPENROUTER_STT_HTTP_RESPONSE", provider="openrouter", model=model, status=response.status_code)
        transcript = _extract_openrouter_transcript(response)
        elapsed = _elapsed_ms(started_at)
        if response.status_code >= 400 or not transcript:
            reason = "http_error" if response.status_code >= 400 else "empty_transcript"
            print(
                f"OPENROUTER_STT_ERROR provider=openrouter model={model} "
                f"status={response.status_code} timingMs={elapsed} transcript=\"{transcript}\" "
                f"fallbackReason={reason}"
            )
            return ""
        log_stage("OPENROUTER_STT_END", provider="openrouter", model=model, status=response.status_code, transcript=transcript)
        print(
            f"STT provider=openrouter status={response.status_code} durationMs={elapsed} transcript=\"{app_log.truncate(transcript)}\""
        )
        return transcript
    except Exception as exc:
        elapsed = _elapsed_ms(started_at) if "started_at" in locals() else 0
        try:
            from .latency import log_stage

            log_stage("OPENROUTER_STT_ERROR", provider="openrouter", model=model, error=exc, fallbackReason="request_error")
        except Exception:
            pass
        print(
            f"OPENROUTER_STT_ERROR provider=openrouter model={model} status=error "
            f"timingMs={elapsed} error={exc} fallbackReason=request_error"
        )
        return ""


def _extract_openrouter_transcript(response):
    try:
        data = response.json()
        if isinstance(data, dict):
            text = data.get("text") or data.get("transcript")
            if text:
                return str(text).strip()
            segments = data.get("segments")
            if isinstance(segments, list):
                return " ".join(str(segment.get("text", "")).strip() for segment in segments if isinstance(segment, dict)).strip()
            return ""
    except Exception as exc:
        print(f"OPENROUTER_STT_ERROR status=parse_error error={exc} fallbackReason=invalid_json")
        return ""
    return ""


def _transcribe_primary(audio, config, allow_remote=True):
    provider = getattr(config, "stt_provider", "local")
    force_local = getattr(config, "use_glasses_mic", False)
    if not allow_remote:
        return transcribe_with_whisper(audio, config, force=force_local)
    if provider == "openrouter":
        return transcribe_with_openrouter(audio, config)
    if provider == "openrouter_first":
        text = transcribe_with_openrouter(audio, config)
        if text:
            return text
        try:
            from .latency import log_stage

            log_stage("LOCAL_STT_FALLBACK_START", provider="faster_whisper", model=getattr(config, "whisper_model", "small.en"))
        except Exception:
            pass
        print(f"LOCAL_STT_FALLBACK provider=faster_whisper model={getattr(config, 'whisper_model', 'small.en')}")
        text = transcribe_with_whisper(audio, config, force=force_local)
        try:
            from .latency import log_stage

            log_stage("LOCAL_STT_FALLBACK_END", provider="faster_whisper", model=getattr(config, "whisper_model", "small.en"), transcript=text)
        except Exception:
            pass
        return text
    if provider == "openai":
        return transcribe_with_openai(audio, config)
    if provider == "openai_first":
        return transcribe_with_openai(audio, config) or transcribe_with_whisper(audio, config)
    return transcribe_with_whisper(audio, config, force=force_local)


def _transcribe_with_google(audio):
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    return recognizer.recognize_google(audio)


def _log_transcription_error(label, exc):
    if exc.__class__.__name__ == "UnknownValueError":
        print(f"{label}: speech was unintelligible")
        return
    print(f"{label}: {exc!r}")
    print(traceback.format_exc().rstrip())


def _trim_filler_prefix(text):
    words = _normalize_text(text).split()
    while words and words[0] in {"uh", "um", "er", "ah", "okay", "ok", "well"}:
        words.pop(0)
    return " ".join(words)


def _token_similarity(left, right):
    left_words = _normalize_text(left).split()
    right_words = _normalize_text(right).split()
    if not left_words or not right_words:
        return 0.0
    left_text = " ".join(left_words)
    right_text = " ".join(right_words)
    ratio = difflib.SequenceMatcher(None, left_text, right_text).ratio()
    overlap = len(set(left_words) & set(right_words)) / max(1, min(len(set(left_words)), len(set(right_words))))
    return max(ratio, overlap)


def _contains_token_sequence(container, needle):
    container_words = _normalize_text(container).split()
    needle_words = _normalize_text(needle).split()
    if not container_words or not needle_words or len(needle_words) > len(container_words):
        return False
    for index in range(0, len(container_words) - len(needle_words) + 1):
        if container_words[index : index + len(needle_words)] == needle_words:
            return True
    return False


def _clean_assistant_echo(transcript):
    from .speech_out import last_spoken_text

    spoken = _normalize_text(last_spoken_text())
    heard = _normalize_text(transcript)
    if not spoken or not heard:
        return transcript
    if heard == spoken or _contains_token_sequence(spoken, heard):
        return ""
    if _contains_token_sequence(heard, spoken) and heard.startswith(spoken):
        return _trim_filler_prefix(heard[len(spoken) :])

    spoken_words = spoken.split()
    heard_words = heard.split()
    max_prefix = min(len(spoken_words), len(heard_words))
    for count in range(max_prefix, 2, -1):
        prefix = " ".join(heard_words[:count])
        spoken_prefix = " ".join(spoken_words[:count])
        if prefix == spoken_prefix or difflib.SequenceMatcher(None, prefix, spoken_prefix).ratio() >= 0.88:
            return _trim_filler_prefix(" ".join(heard_words[count:]))
    return transcript


def _reject_assistant_echo(transcript, threshold=0.82):
    from .speech_out import last_spoken_text

    spoken = last_spoken_text()
    heard = _normalize_text(transcript)
    if not spoken or not heard:
        return False
    normalized_spoken = _normalize_text(spoken)
    if normalized_spoken == heard or _contains_token_sequence(heard, normalized_spoken) or _contains_token_sequence(normalized_spoken, heard):
        similarity = 1.0
    else:
        similarity = _token_similarity(heard, spoken)
    if similarity >= threshold:
        print(f"REPLY_TRANSCRIPT_REJECTED reason=assistant_echo similarity={similarity:.2f}")
        return True
    return False


def _record_glasses_audio(config, phrase_time_limit, quiet=False, label="recording", fixed_duration=False, silence_ms=None, latency_prefix=None):
    from .mic_ingest import glasses_mic_buffer

    seconds = float(phrase_time_limit)
    timeout = float(getattr(config, "mic_listen_timeout", 12))
    if not quiet:
        print(f"{label} from glasses mic for {seconds:.1f} seconds...")
        print(f"AUDIO_CAPTURE_START source=glasses label={label} maxSeconds={seconds:.1f}")
    if latency_prefix and latency_prefix != "WAKE_AUDIO_CAPTURE":
        from .latency import log_stage

        log_stage(_latency_stage(latency_prefix, "start"), source="glasses", maxSeconds=seconds)
    started_at = time.monotonic()
    buffer = glasses_mic_buffer()
    if fixed_duration:
        pcm = buffer.read_fresh_seconds(seconds, timeout=timeout)
    else:
        buffer.clear()
        chunk_seconds = 0.05
        chunk_bytes = int(16000 * 2 * chunk_seconds)
        silence_limit = max(1, int(_silence_seconds(config, silence_ms) / chunk_seconds))
        silence_rms = _wake_dynamic_threshold(config) if latency_prefix == "WAKE_AUDIO_CAPTURE" else _glasses_silence_rms(seconds, fixed_duration)
        chunks = []
        pre_roll = deque(maxlen=max(1, int(0.3 / chunk_seconds)))
        speech_started = False
        consecutive_voiced = 0
        silent_chunks = 0
        voiced_chunks = 0
        rms_values = []
        all_rms_values = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            chunk = buffer.read_bytes(chunk_bytes, timeout=min(0.15, max(0.01, deadline - time.monotonic())))
            if not chunk:
                continue
            if len(chunk) % 2:
                chunk = chunk[:-1]
            rms = audioop.rms(chunk, 2) if chunk else 0
            all_rms_values.append(rms)
            if rms >= silence_rms:
                if not speech_started:
                    consecutive_voiced += 1
                    pre_roll.append(chunk)
                    if consecutive_voiced < 2:
                        continue
                if not speech_started and not quiet:
                    print(f"SPEECH_STARTED source=glasses timingMs={_elapsed_ms(started_at)}")
                if not speech_started:
                    if latency_prefix:
                        if latency_prefix == "WAKE_AUDIO_CAPTURE":
                            from .latency import log_stage, new_interaction

                            new_interaction()
                            log_stage("WAKE_LISTEN_START", source="glasses", maxSeconds=seconds)
                        log_stage(_latency_stage(latency_prefix, "speech"), source="glasses", rms=rms)
                    chunks.extend(pre_roll)
                speech_started = True
                silent_chunks = 0
                voiced_chunks += 1
                rms_values.append(rms)
            elif speech_started:
                silent_chunks += 1
            elif not speech_started:
                consecutive_voiced = 0
                pre_roll.append(chunk)
            if speech_started:
                chunks.append(chunk)
            if speech_started and silent_chunks >= silence_limit:
                if not quiet:
                    print(f"SILENCE_DETECTED source=glasses timingMs={_elapsed_ms(started_at)}")
                if latency_prefix:
                    log_stage(_latency_stage(latency_prefix, "silence"), source="glasses")
                break
        pcm = b"".join(chunks)
        if latency_prefix == "WAKE_AUDIO_CAPTURE" and not pcm:
            from .latency import log_stage

            peak_rms = max(all_rms_values) if all_rms_values else 0
            _log_wake_audio_rejected("below_dynamic_threshold", peak_rms, silence_rms)
            log_stage("WAKE_AUDIO_REJECTED", reason="below_dynamic_threshold", rms=peak_rms, threshold=silence_rms)
            return ""
        if latency_prefix == "WAKE_AUDIO_CAPTURE":
            voiced_ms = int(voiced_chunks * chunk_seconds * 1000)
            peak_rms = max(rms_values) if rms_values else 0
            avg_rms = int(sum(rms_values) / len(rms_values)) if rms_values else 0
            min_voiced = int(getattr(config, "wake_min_voiced_ms", 250))
            if peak_rms < silence_rms or voiced_ms < min_voiced or len(pcm) < int(16000 * 2 * 0.25):
                from .latency import log_stage

                reason = "below_dynamic_threshold" if peak_rms < silence_rms else "too_short"
                _log_wake_audio_rejected(reason, peak_rms, silence_rms)
                log_stage("WAKE_AUDIO_REJECTED", reason=reason, rms=peak_rms, avgRms=avg_rms, threshold=silence_rms, voicedMs=voiced_ms)
                return ""
    if not quiet:
        print(f"AUDIO_CAPTURE_END source=glasses timingMs={_elapsed_ms(started_at)} bytes={len(pcm) if pcm else 0}")
    if latency_prefix:
        log_stage(_latency_stage(latency_prefix, "end"), source="glasses", bytes=len(pcm) if pcm else 0)
    if not pcm:
        if not quiet:
            print("Glasses mic: no audio yet — keep Glasses -> Start everything running on the phone.")
        return ""
    if len(pcm) % 2:
        pcm = pcm[:-1]
    rms = audioop.rms(pcm, 2) if pcm else 0
    silence_rms = _wake_dynamic_threshold(config) if latency_prefix == "WAKE_AUDIO_CAPTURE" else _glasses_silence_rms(seconds, fixed_duration)
    if latency_prefix != "WAKE_AUDIO_CAPTURE" and rms < silence_rms:
        if not quiet:
            print(f"Glasses mic: ignoring silence (rms={rms})")
        return ""
    if not quiet:
        print(f"glasses mic: captured audio chunk (rms={rms})")
    try:
        import speech_recognition as sr

        return sr.AudioData(pcm, 16000, 2)
    except Exception as exc:
        print(f"Glasses mic capture failed: {exc}")
        return ""


def _record_audio(
    config,
    phrase_time_limit,
    typed_fallback=True,
    quiet=False,
    label="recording",
    mic_index=None,
    fixed_duration=False,
    after_tts=False,
    silence_ms=None,
    latency_prefix=None,
):
    _wait_after_tts(config, after_tts)
    if getattr(config, "use_glasses_mic", False):
        return _record_glasses_audio(config, phrase_time_limit, quiet, label, fixed_duration, silence_ms=silence_ms, latency_prefix=latency_prefix)
    with _audio_lock:
        last_error = None
        for attempt in range(2):
            try:
                import speech_recognition as sr

                recognizer = sr.Recognizer()
                recognizer.pause_threshold = _silence_seconds(config, silence_ms)
                device_index = getattr(config, "mic_device_index", 0) if mic_index is None else mic_index
                names = sr.Microphone.list_microphone_names()
                if 0 <= device_index < len(names) and not quiet:
                    print(f"using microphone: {names[device_index]}")
                with sr.Microphone(device_index=device_index) as source:
                    timeout = phrase_time_limit if not typed_fallback else getattr(config, "mic_listen_timeout", 12)
                    if not quiet:
                        print(f"{label} for {phrase_time_limit} seconds...")
                        print(f"AUDIO_CAPTURE_START source=mac_mic label={label} maxSeconds={float(phrase_time_limit):.1f}")
                    if latency_prefix:
                        from .latency import log_stage

                        log_stage(_latency_stage(latency_prefix, "start"), source="mac_mic", maxSeconds=float(phrase_time_limit))
                    started_at = time.monotonic()
                    _prepare_microphone(recognizer, source, config)
                    if not quiet:
                        print("listening now...")
                    start_pause = float(getattr(config, "mic_phrase_start_pause", 0.2))
                    if start_pause > 0:
                        time.sleep(start_pause)
                    if fixed_duration:
                        audio = recognizer.record(source, duration=phrase_time_limit)
                    else:
                        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                if not quiet:
                    print(f"AUDIO_CAPTURE_END source=mac_mic timingMs={_elapsed_ms(started_at)}")
                if latency_prefix:
                    log_stage(_latency_stage(latency_prefix, "end"), source="mac_mic")
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


def listen(config=None, typed_fallback=True, record_seconds=None, label="recording", after_tts=False, silence_ms=None):
    short_mode = bool(getattr(config, "whisper_short_reply_mode", True))
    record_seconds = float(record_seconds if record_seconds is not None else getattr(config, "whisper_record_seconds", 5))
    started_at = time.monotonic()

    prefix = "FOLLOWUP_CAPTURE" if "follow" in label.lower() else "COMMAND_CAPTURE"
    transcript = ""
    for attempt in range(2):
        audio = _record_audio(
            config,
            record_seconds,
            typed_fallback,
            quiet=False,
            label=label,
            after_tts=after_tts if attempt == 0 else False,
            silence_ms=silence_ms,
            latency_prefix=prefix if attempt == 0 else None,
        )
        if isinstance(audio, str):
            transcript = audio
            break

        raw_transcript = _transcribe_with_fallback(audio, config, short_mode, typed_fallback, label, after_tts=after_tts, silence_ms=silence_ms)
        cleaned = _clean_assistant_echo(raw_transcript)
        if cleaned:
            transcript = cleaned
        elif raw_transcript and _reject_assistant_echo(raw_transcript):
            transcript = ""
        else:
            transcript = cleaned
        if transcript and not _reject_assistant_echo(transcript):
            break
        if raw_transcript and attempt == 0:
            transcript = ""
            continue
        break
    print(f"TOTAL_REPLY_TRANSCRIPTION_MS timingMs={_elapsed_ms(started_at)} transcript=\"{transcript}\"")
    try:
        from .latency import log_stage

        log_stage("TRANSCRIPT_READY", transcript=transcript)
    except Exception:
        pass
    return transcript


def _transcribe_with_fallback(audio, config, short_mode, typed_fallback, label, after_tts=False, silence_ms=None):
    glasses_mic = bool(getattr(config, "use_glasses_mic", False))
    ok, _stats = reply_audio_is_usable(audio, config)
    if not ok:
        return ""
    text = _transcribe_primary(audio, config, allow_remote=True)
    if text:
        return text

    try:
        text = _transcribe_with_google(audio)
        if text:
            print("STT: Google fallback succeeded on first recording")
            return text
    except Exception as exc:
        _log_transcription_error("Google STT fallback failed", exc)

    if short_mode:
        print("short reply retry...")
        audio = _record_audio(config, 3, typed_fallback, quiet=False, label=label, after_tts=after_tts, silence_ms=silence_ms)
        if isinstance(audio, str):
            return audio
        text = _transcribe_primary(audio, config, allow_remote=True)
        if text:
            return text

    try:
        return _transcribe_with_google(audio)
    except Exception as exc:
        _log_transcription_error("Google STT final fallback failed", exc)
        if glasses_mic or not typed_fallback:
            return ""
        print(f"Microphone speech input unavailable: {exc}")
        print("Type your response instead, or allow microphone permission on macOS and install PyAudio.")
        try:
            return input("> ")
        except EOFError:
            return ""


def _wake_capture_blocked(state):
    from .speech_out import is_speaking

    if is_speaking():
        return True
    if state is None:
        return False
    return bool(
        getattr(state, "is_busy", False)
        or getattr(state, "is_processing_command", False)
        or getattr(state, "awaiting_follow_up_reply", False)
        or getattr(state, "follow_up_enabled", False)
    )


def _is_assistant_echo(transcript):
    from .speech_out import last_spoken_text

    spoken = _normalize_text(last_spoken_text())
    heard = _normalize_text(transcript)
    if not spoken or not heard:
        return False
    if _contains_token_sequence(spoken, heard) or _contains_token_sequence(heard, spoken):
        return True
    heard_words = heard.split()
    if len(heard_words) < 3:
        return False
    return _token_similarity(heard, spoken) >= 0.82


def listen_for_wake_phrase(mic_index=None, config=None, state=None):
    if config is None:
        return None
    if not getattr(config, "voice_activation_mode", True) and not getattr(config, "wake_mode", False):
        return None
    if _wake_capture_blocked(state):
        return None
    from .latency import clear_interaction, current_interaction_id, log_stage, snapshot

    clear_interaction()
    app_log.debug("wake listening...")
    audio = _record_audio(
        config,
        float(getattr(config, "wake_capture_seconds", 2.0)),
        typed_fallback=False,
        quiet=True,
        label="wake recording",
        mic_index=mic_index,
        fixed_duration=False,
        silence_ms=getattr(config, "wake_silence_ms", 400),
        latency_prefix="WAKE_AUDIO_CAPTURE",
    )
    if isinstance(audio, str) or _wake_capture_blocked(state):
        return None

    log_stage("WAKE_TRANSCRIPTION_START")
    wake_text = transcribe_wake_with_whisper(audio, config)
    if wake_text is None:
        log_stage("WAKE_TRANSCRIPTION_END", transcript="", model=getattr(config, "wake_whisper_model", "tiny.en"), abandoned=True)
        log_stage("WAKE_PHRASE_REJECTED", reason="wake_transcription_timeout")
        return None
    transcript = _clean_assistant_echo(wake_text)
    log_stage("WAKE_TRANSCRIPTION_END", transcript=transcript, model=getattr(config, "wake_whisper_model", "tiny.en"))
    rejection = wake_transcript_rejection(transcript)
    if rejection:
        print(
            "WAKE_TRANSCRIPT_REJECTED "
            f"reason={rejection['reason']} tokens={rejection['tokens']} "
            f"unique={rejection['unique']} preview=\"{rejection['preview']}\""
        )
        log_stage("WAKE_TRANSCRIPT_REJECTED", **rejection)
        return None
    normalized = _normalize_text(transcript)
    if not normalized:
        try:
            google_transcript = _transcribe_with_google(audio)
            if google_transcript:
                transcript = _clean_assistant_echo(google_transcript)
                normalized = _normalize_text(transcript)
        except Exception as exc:
            _log_transcription_error("Wake Google STT unavailable", exc)
    elif not wake_phrase_detected(normalized, config):
        try:
            google_transcript = _transcribe_with_google(audio)
            if google_transcript and _normalize_text(google_transcript):
                alt = _normalize_text(google_transcript)
                if wake_phrase_detected(alt, config):
                    transcript = _clean_assistant_echo(google_transcript)
                    normalized = _normalize_text(transcript)
        except Exception as exc:
            _log_transcription_error("Wake Google STT fallback failed", exc)

    debug = bool(getattr(config, "wake_debug_transcripts", True))
    log_empty = bool(getattr(config, "wake_log_empty_transcripts", False))
    if not normalized:
        if debug and log_empty:
            print('wake heard: ""')
            print("wake phrase not detected")
            log_stage("WAKE_PHRASE_REJECTED", reason="empty")
        return None

    if not is_wake_check_command(normalized) and _is_assistant_echo(normalized):
        app_log.debug(f'wake rejected: assistant echo ("{_preview(normalized)}")')
        log_stage("WAKE_PHRASE_REJECTED", reason="assistant_echo", transcript=normalized)
        return None

    if debug or normalized:
        app_log.debug(f'wake heard: "{_preview(normalized)}"')

    if wake_phrase_detected(normalized, config):
        print(f'WAKE accepted transcript="{_preview(normalized)}"')
        log_stage("WAKE_PHRASE_ACCEPTED", transcript=normalized)
        return {"transcript": transcript, **snapshot()}

    if getattr(state, "follow_up_enabled", False) or getattr(state, "awaiting_follow_up_reply", False):
        print(f'follow-up speech captured: "{normalized}"')
        log_stage("WAKE_PHRASE_ACCEPTED", transcript=normalized, followup=True)
        return {"transcript": transcript, **snapshot()}

    reason = _wake_rejection_reason(normalized, config)
    if reason != "not detected":
        app_log.debug(f"wake rejected: {reason}")
    else:
        app_log.debug("wake phrase not detected")
    log_stage("WAKE_PHRASE_REJECTED", reason=reason, transcript=normalized)
    return None


def _demo():
    class C:
        wake_phrases = ("hey look", "okay look", "hey assistant", "hey glasses", "hey nova", "nova")
        allow_single_word_wake = False
        wake_debug_transcripts = True
        wake_log_empty_transcripts = False
        wake_min_transcript_length = 2

    assert strip_wake_phrase("Hey Look, what is this?", C.wake_phrases, False) == "what is this"
    assert strip_wake_phrase("hey look can you read a sign", C.wake_phrases, False) == "can you read a sign"
    assert strip_wake_phrase("hey look", C.wake_phrases, False) == ""
    assert wake_phrase_detected("hey look what is this", C())
    assert wake_phrase_detected("hay look read this sign", C())
    assert wake_phrase_detected("okay look what is that", C())
    assert strip_wake_phrase("hey trails can you tell me what this is", ("hey trail",), False) == "can you tell me what this is"
    assert wake_phrase_detected("hey trail what is this", type("C", (), {"wake_phrases": ("hey trail",), "allow_single_word_wake": False})())
    assert wake_phrase_detected("hey nova what is this", C())
    assert wake_phrase_detected("nova, what is this", C())
    assert wake_phrase_detected("tell can you hear me", C())
    assert is_wake_check_command("can you hear me")
    assert not wake_phrase_detected("hair trail", C())
    assert not wake_phrase_detected("trail", C())
    assert is_incomplete_command("can you")
    assert is_incomplete_command("what is the")
    assert direct_voice_command_detected("can you tell me what this is", C())
    assert direct_voice_command_detected("can you read a sign", C())
    assert direct_voice_command_detected("what is this", C())
    assert not direct_voice_command_detected("hello there", C())
    import src.speech_out as speech_out
    speech_out._last_spoken_text = "I can take a closer look at this sign if you want."
    assert _is_assistant_echo("i can take a closer look at")
    speech_out._last_spoken_text = "Should I explain it?"
    assert _clean_assistant_echo("Should I explain it? Uh, no") == "no"
    assert _clean_assistant_echo("Should I explain it?") == ""


if __name__ == "__main__":
    _demo()
