import os
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid

from .audio_http import enqueue_glasses_speech, prepare_glasses_speech, public_tts_url, _log_tts
from .latency import log_stage, set_event


_speaking = threading.Event()
_last_spoken_text = ""
_last_spoken_at = 0.0
_playback_lock = threading.Lock()
_active_playback = None
_awaiting_first_trail_transcript = threading.Event()


def _playback_timeout_seconds():
    try:
        return max(2.0, float(os.getenv("TTS_PLAYBACK_MAX_SECONDS", "10")))
    except ValueError:
        return 10.0


def _estimated_playback_seconds(text):
    words = len((text or "").split())
    return min(2.5, max(0.8, (words / 3.0) + 0.25))


def _post_playback_guard_seconds():
    try:
        return min(0.25, max(0.15, float(os.getenv("POST_PLAYBACK_GUARD_SECONDS", "0.2"))))
    except ValueError:
        return 0.2


def _clear_tts_state(event_id, reason):
    global _active_playback
    with _playback_lock:
        active = _active_playback
        if not active or active["eventId"] != event_id:
            return
        for timer in active["timers"]:
            timer.cancel()
        _active_playback = None
        _speaking.clear()

    from . import app_log

    app_log.debug(f"TTS_STATE cleared eventId={event_id} reason={reason}")
    try:
        from .mic_ingest import glasses_mic_buffer

        mic_buffer = glasses_mic_buffer()
    except Exception as exc:
        mic_buffer = None
        print(f"TTS microphone buffer unavailable: {exc}")
    if mic_buffer is not None:
        try:
            mic_buffer.clear()
        except Exception as exc:
            print(f"TTS microphone pre-delay cleanup failed: {exc}")
    expected_at = active.get("replyCaptureExpectedAt", time.monotonic())
    delay_ms = int((time.monotonic() - expected_at) * 1000)
    print(f"REPLY_CAPTURE started delayFromExpectedMs={delay_ms}")
    if mic_buffer is not None:
        try:
            mic_buffer.clear()
        except Exception as exc:
            print(f"TTS microphone post-delay cleanup failed: {exc}")
    if active["trailCommand"]:
        print(f"TRAIL_COMMAND_TTS_END eventId={event_id} reason={reason}")
    log_stage("TTS_STATE_CLEARED", event_id=event_id, reason=reason)
    log_stage("WAKE_LISTENER_RESUMED", event_id=event_id, source="tts_state")
    app_log.debug(f"WAKE_LISTENER_RESUMED eventId={event_id} source=tts_state")


def _schedule_clear(event_id, delay, reason):
    timer = threading.Timer(delay, _clear_tts_state, args=(event_id, reason))
    timer.daemon = True
    with _playback_lock:
        if not _active_playback or _active_playback["eventId"] != event_id:
            return
        _active_playback["timers"].append(timer)
    timer.start()


def _begin_playback(event_id, text, trail_command):
    global _active_playback
    with _playback_lock:
        previous = _active_playback
        if previous:
            for timer in previous["timers"]:
                timer.cancel()
        _active_playback = {
            "eventId": event_id,
            "text": text,
            "trailCommand": trail_command,
            "timers": [],
            "audioDurationMs": 0,
            "replyCaptureExpectedAt": 0.0,
        }
        _speaking.set()
    if trail_command:
        _awaiting_first_trail_transcript.set()
        print(f"TRAIL_COMMAND_TTS_START eventId={event_id}")
    _schedule_clear(event_id, _playback_timeout_seconds(), "playback_timeout")


def notify_playback_started(event_id):
    with _playback_lock:
        active = _active_playback
        if not active or active["eventId"] != event_id:
            return
        text = active["text"]
        audio_duration_ms = int(active.get("audioDurationMs") or 0)
    guard = _post_playback_guard_seconds()
    delay = (audio_duration_ms / 1000.0 if audio_duration_ms > 0 else _estimated_playback_seconds(text)) + guard
    with _playback_lock:
        if _active_playback and _active_playback["eventId"] == event_id:
            _active_playback["replyCaptureExpectedAt"] = time.monotonic() + delay
    from . import app_log

    app_log.debug(f"PLAYBACK_DURATION_MS eventId={event_id} value={audio_duration_ms}")
    print(f"REPLY_CAPTURE scheduledInMs={int(delay * 1000)} audioDurationMs={audio_duration_ms} guardMs={int(guard * 1000)}")
    _schedule_clear(event_id, delay, "playback_duration_complete")
    log_stage("BLUETOOTH_PLAYBACK_START", event_id=event_id, inferred=False)


def set_playback_duration(event_id, audio_duration_ms):
    with _playback_lock:
        if _active_playback and _active_playback["eventId"] == event_id:
            _active_playback["audioDurationMs"] = int(audio_duration_ms or 0)


def log_first_transcript_after_trail(transcript):
    if transcript and _awaiting_first_trail_transcript.is_set():
        _awaiting_first_trail_transcript.clear()
        print(f'first transcript after trail command: "{transcript}"')


def _glasses_audio_enabled():
    return bool((os.getenv("ANDROID_TRAIL_URL", "") or "").strip())


def _mirror_to_android_glasses(text, event_id="", audio_url="", phrase_id=""):
    base = (os.getenv("ANDROID_TRAIL_URL", "") or "").strip().rstrip("/")
    if not base or not text:
        return False
    try:
        log_stage("ANDROID_SPEAK_REQUEST_START", event_id=event_id, endpoint="/trail/speak")
        query = urllib.parse.urlencode({"text": text, "phraseId": phrase_id, "eventId": event_id, "audioUrl": audio_url})
        request = urllib.request.Request(
            f"{base}/trail/speak?{query}",
            method="POST",
            data=b"",
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            ok = 200 <= response.status < 300
            if ok:
                print("glasses audio: sent to phone trail server for Bluetooth playback")
                _log_tts(event_id, "android_command_received", endpoint="/trail/speak", status=response.status)
                log_stage("ANDROID_SPEAK_REQUEST_END", event_id=event_id, endpoint="/trail/speak", status=response.status)
            return ok
    except Exception as exc:
        log_stage("ANDROID_SPEAK_REQUEST_END", event_id=event_id, endpoint="/trail/speak", error=exc)
        print(f"glasses audio: phone trail server unavailable ({exc})")
        return False


def is_speaking():
    return _speaking.is_set()


def last_spoken_text(max_age_seconds=12.0):
    if max_age_seconds and _last_spoken_at and time.monotonic() - _last_spoken_at > max_age_seconds:
        return ""
    return _last_spoken_text


def speak(text, trail_command=False):
    global _last_spoken_text, _last_spoken_at
    _last_spoken_text = (text or "").strip()
    _last_spoken_at = time.monotonic()
    event_id = uuid.uuid4().hex[:8]
    set_event(event_id)
    playback_deferred = False
    print(f'ASSISTANT "{_last_spoken_text}"')
    log_stage("RESPONSE_TEXT_READY", event_id=event_id, textBytes=len(_last_spoken_text.encode("utf-8")))
    _log_tts(event_id, "response_text_ready", textBytes=len(_last_spoken_text.encode("utf-8")))
    _begin_playback(event_id, _last_spoken_text, trail_command)
    try:
        if _glasses_audio_enabled():
            record = {}
            try:
                record = prepare_glasses_speech(_last_spoken_text, event_id)
                set_playback_duration(event_id, record.get("audioDurationMs", 0))
            except Exception as exc:
                print(f"TTS prebuild failed: {exc}")
            audio_url = public_tts_url(event_id)
            phrase_id = record.get("phraseId", "") if record else ""
            if _mirror_to_android_glasses(_last_spoken_text, event_id=event_id, audio_url=audio_url, phrase_id=phrase_id):
                playback_deferred = True
                print(f'SPEECH_OUT\ntarget=ANDROID\neventId={event_id}\ntext="{_last_spoken_text}"')
                return
            enqueue_glasses_speech(_last_spoken_text, event_id=event_id, audio_url=audio_url)
            playback_deferred = True
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
        if not playback_deferred:
            _clear_tts_state(event_id, "speak_complete")
