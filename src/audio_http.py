from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import threading
import time
import wave
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .config import Config, EXPECTED_MAC_IP
from .common_tts import AUDIO_DIR, lookup_by_phrase_id, lookup_by_text, template_match, upsert_generated_template
from . import app_log
from .intent import Intent, classify_intent_with_source
from .latency import log_stage, summarize_event
from .llm import answer_for, answer_general_question_with_1b, describe_current_object
from .phrases import get_cancel_response, get_clarification_response, get_repeat_response
from .session_state import SessionState

_pending_speech: deque[dict] = deque(maxlen=32)
_pending_trail: deque[str] = deque(maxlen=16)
_pending_lock = threading.Lock()
_runtime_status = {"androidBridge": False, "mentraStream": False, "video": False, "audio": False, "whisper": False}
_tts_cache: dict[str, dict] = {}
_tts_cache_lock = threading.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _log_tts(event_id: str, phase: str, **values):
    if not app_log.enabled("DEBUG"):
        return
    parts = [f"TTS_TIMING\nphase={phase}", f"eventId={event_id}", f"atMs={_now_ms()}"]
    parts.extend(f"{key}={value}" for key, value in values.items())
    print("\n".join(parts))


def wav_duration_ms(wav: bytes) -> int:
    try:
        import io

        with wave.open(io.BytesIO(wav), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return int(frames / max(1, rate) * 1000)
    except Exception:
        return 0


def update_runtime_status(**values):
    _runtime_status.update(values)


def _default_tts_base_url() -> str:
    return f"http://{EXPECTED_MAC_IP}:8765"


def public_tts_url(event_id: str) -> str:
    base = _coerce_text(os.getenv("MAC_TTS_BASE_URL") or _default_tts_base_url()).rstrip("/")
    return f"{base}/tts?eventId={event_id}"


def public_phrase_url(phrase_id: str) -> str:
    base = _coerce_text(os.getenv("MAC_TTS_BASE_URL") or _default_tts_base_url()).rstrip("/")
    return f"{base}/tts?phraseId={phrase_id}"


def _record_from_common(entry: dict, event_id: str, text_ready_at_ms: int) -> dict:
    wav = Path(entry["path"]).read_bytes()
    return {
        "eventId": event_id,
        "phraseId": entry.get("phraseId", ""),
        "text": entry.get("text", ""),
        "wav": wav,
        "sampleRate": 16000,
        "channels": 1,
        "textReadyAtMs": text_ready_at_ms,
        "createdAtMs": _now_ms(),
        "generationMs": 0,
        "audioDurationMs": wav_duration_ms(wav),
        "wavPath": str(entry["path"]),
    }


def prepare_glasses_speech(text: str, event_id: str, sample_rate: int = 16000, channels: int = 1) -> dict:
    cleaned = (text or "").strip()
    if not cleaned:
        return {}
    text_ready_at_ms = _now_ms()
    with _tts_cache_lock:
        cached = _tts_cache.get(event_id)
        if cached:
            return cached

    log_stage("COMMON_AUDIO_LOOKUP_START", event_id=event_id)
    common = lookup_by_text(cleaned)
    if common:
        log_stage("COMMON_AUDIO_HIT", event_id=event_id, phraseId=common.get("phraseId", ""))
        record = _record_from_common(common, event_id, text_ready_at_ms)
        with _tts_cache_lock:
            _tts_cache[event_id] = record
        return record

    log_stage("COMMON_AUDIO_MISS", event_id=event_id)
    template = template_match(cleaned)
    if template:
        wav_path = AUDIO_DIR / template["wav"]
        if not wav_path.exists():
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            log_stage("TTS_GENERATION_START", event_id=event_id, sampleRate=sample_rate, channels=channels, templateId=template["templateId"])
            _log_tts(event_id, "tts_generation_start", sampleRate=sample_rate, channels=channels, textBytes=len(cleaned.encode("utf-8")), templateId=template["templateId"])
            wav_path.write_bytes(synthesize_wav(cleaned, sample_rate=sample_rate, channels=channels))
            log_stage("TTS_GENERATION_END", event_id=event_id, audioBytes=wav_path.stat().st_size, templateId=template["templateId"])
            _log_tts(event_id, "tts_generation_end", generationMs=int((time.monotonic() - started) * 1000), audioBytes=wav_path.stat().st_size, templateId=template["templateId"])
        upsert_generated_template(template)
        common = {**template, "path": wav_path}
        app_log.info(f"COMMON_AUDIO_TEMPLATE_HIT phraseId={template['phraseId']} lookupMs=0 ttsGenerationMs=0")
        record = _record_from_common(common, event_id, text_ready_at_ms)
        with _tts_cache_lock:
            _tts_cache[event_id] = record
        return record

    started = time.monotonic()
    log_stage("TTS_GENERATION_START", event_id=event_id, sampleRate=sample_rate, channels=channels)
    _log_tts(event_id, "tts_generation_start", sampleRate=sample_rate, channels=channels, textBytes=len(cleaned.encode("utf-8")))
    wav = synthesize_wav(cleaned, sample_rate=sample_rate, channels=channels)
    record = {
        "eventId": event_id,
        "phraseId": "",
        "text": cleaned,
        "wav": wav,
        "sampleRate": sample_rate,
        "channels": channels,
        "textReadyAtMs": text_ready_at_ms,
        "createdAtMs": _now_ms(),
        "generationMs": int((time.monotonic() - started) * 1000),
        "audioDurationMs": wav_duration_ms(wav),
    }
    with _tts_cache_lock:
        _tts_cache[event_id] = record
        while len(_tts_cache) > 32:
            _tts_cache.pop(next(iter(_tts_cache)))
    log_stage("TTS_GENERATION_END", event_id=event_id, audioBytes=len(wav))
    _log_tts(event_id, "tts_generation_end", generationMs=record["generationMs"], audioBytes=len(wav))
    return record


def enqueue_glasses_speech(text: str, event_id: str | None = None, audio_url: str | None = None) -> None:
    cleaned = (text or "").strip()
    if not cleaned:
        return
    event_id = event_id or str(_now_ms())
    with _tts_cache_lock:
        phrase_id = (_tts_cache.get(event_id) or {}).get("phraseId", "")
    with _pending_lock:
        url = audio_url or public_tts_url(event_id)
        _pending_speech.append({"text": cleaned, "phraseId": phrase_id, "eventId": event_id, "audio_url": url, "audioUrl": url})


def dequeue_glasses_speech() -> dict:
    with _pending_lock:
        if _pending_speech:
            return _pending_speech.popleft()
    return {}


def enqueue_trail_command(action: str) -> None:
    cleaned = (action or "").strip()
    if not cleaned:
        return
    with _pending_lock:
        _pending_trail.append(cleaned)


def dequeue_trail_command() -> str:
    with _pending_lock:
        if _pending_trail:
            return _pending_trail.popleft()
    return ""

IMAGE_INTENTS = {
    Intent.EXPLAIN_CURRENT_OBJECT,
    Intent.READ_SIGN_TEXT,
    Intent.EXPLAIN_SIGN_MEANING,
    Intent.IDENTIFY_PLANT,
    Intent.EXPLAIN_PLANT,
    Intent.DESCRIBE_CURRENT_OBJECT,
    Intent.IDENTIFY_CURRENT_OBJECT,
    Intent.WHAT_AM_I_LOOKING_AT,
}


def synthesize_wav(text: str, sample_rate: int = 16000, channels: int = 1) -> bytes:
    text = (text or "").strip()
    with tempfile.TemporaryDirectory(prefix="mentra-tts-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        aiff_path = tmpdir_path / "tts.aiff"
        wav_path = tmpdir_path / "tts.wav"
        subprocess.run(["say", "-o", str(aiff_path), text], check=True)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(aiff_path),
                "-ac",
                str(channels),
                "-ar",
                str(sample_rate),
                "-acodec",
                "pcm_s16le",
                str(wav_path),
            ],
            check=True,
        )
        return wav_path.read_bytes()


def _decode_image_base64(image_b64: str):
    raw = base64.b64decode(image_b64)
    data = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image_base64")
    return image


def _payload_from_query(query: dict[str, list[str]]) -> dict:
    payload: dict = {}
    for key, values in query.items():
        if not values:
            continue
        payload[key] = values[0]
    return payload


def _payload_from_body(body: bytes) -> dict:
    if not body:
        return {}
    try:
        parsed = json.loads(body.decode("utf-8"))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed_qs = parse_qs(body.decode("utf-8"))
        return _payload_from_query(parsed_qs)
    except Exception:
        return {}


def _coerce_text(value) -> str:
    return ("" if value is None else str(value)).strip()


def _assistant_answer_from_payload(payload: dict, config: Config) -> tuple[str, str, str]:
    mode = _coerce_text(payload.get("mode") or payload.get("request_mode") or "assistant").lower()
    text = _coerce_text(payload.get("text") or payload.get("query") or payload.get("prompt"))
    kind = _coerce_text(payload.get("kind") or payload.get("detected_kind") or "") or None
    last_question_type = _coerce_text(payload.get("last_question_type") or "none") or "none"
    last_message = _coerce_text(payload.get("last_message") or payload.get("assistant_prompt") or "")
    ocr_text = _coerce_text(payload.get("ocr_text") or "")
    image_b64 = _coerce_text(payload.get("image_base64") or payload.get("image") or "")

    if mode == "tts":
        return text or "", "tts", "direct"

    if image_b64:
        crop = _decode_image_base64(image_b64)
        state = SessionState()
        if kind in {"sign", "plant"}:
            intent_name = _coerce_text(payload.get("intent") or payload.get("action") or "")
            try:
                intent = Intent(intent_name) if intent_name else Intent.WHAT_AM_I_LOOKING_AT
            except Exception:
                intent = Intent.WHAT_AM_I_LOOKING_AT
            if intent == Intent.EXPLAIN_CURRENT_OBJECT:
                intent = Intent.WHAT_AM_I_LOOKING_AT
            answer = answer_for(kind, crop, intent, config, state)
            return answer, intent.value, kind

        answer = describe_current_object(crop, config, state)
        return answer, Intent.WHAT_AM_I_LOOKING_AT.value, kind or state.last_detected_kind or "object"

    if not text:
        return get_clarification_response(), Intent.ASK_CLARIFICATION.value, kind or "unknown"

    intent, source = classify_intent_with_source(
        text,
        config,
        kind,
        last_message,
        ocr_text,
        last_question_type=last_question_type,
    )

    if intent == Intent.GENERAL_QUESTION:
        answer = answer_general_question_with_1b(text, config)
    elif intent in {Intent.CANCEL, Intent.STOP_LISTENING}:
        answer = get_cancel_response()
    elif intent == Intent.REPEAT_LAST_MESSAGE:
        answer = get_repeat_response()
    elif intent == Intent.SPEAK_SLOWER:
        answer = "Of course. I’ll keep it slower and brief."
    elif intent in IMAGE_INTENTS:
        answer = "Please send a camera frame or photo so I can look at that."
    elif intent == Intent.MORE_DETAIL:
        answer = "Please send the current view first, and I can give more detail."
    else:
        answer = get_clarification_response()

    return answer, intent.value, source


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _BackendHandler(BaseHTTPRequestHandler):
    server_version = "MentraBackend/1.0"

    def do_GET(self):
        self._handle_request(is_post=False)

    def do_POST(self):
        self._handle_request(is_post=True)

    def _handle_request(self, is_post: bool):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/health"}:
            _json_response(self, 200, {"ok": True, **_runtime_status})
            return

        if parsed.path == "/pending":
            item = dequeue_glasses_speech()
            if item:
                _log_tts(item.get("eventId", ""), "android_command_received", endpoint="/pending")
                log_stage("ANDROID_SPEAK_REQUEST_END", event_id=item.get("eventId", ""), endpoint="/pending")
            _json_response(self, 200, {"text": "", **item})
            return

        if parsed.path == "/pending-trail":
            action = dequeue_trail_command()
            _json_response(self, 200, {"action": action})
            return

        if parsed.path == "/tts":
            query = parse_qs(parsed.query)
            event_id = _coerce_text(query.get("eventId", query.get("event_id", [""]))[0])
            phrase_id = _coerce_text(query.get("phraseId", query.get("phrase_id", [""]))[0])
            text = _coerce_text(query.get("text", [""])[0])
            rate = int(_coerce_text(query.get("rate", ["16000"])[0]) or "16000")
            channels = int(_coerce_text(query.get("channels", ["1"])[0]) or "1")
            self._send_wav(text, rate, channels, event_id=event_id, phrase_id=phrase_id)
            return

        if parsed.path == "/playback-start":
            query = parse_qs(parsed.query)
            event_id = _coerce_text(query.get("eventId", query.get("event_id", [""]))[0])
            _log_tts(event_id, "bluetooth_playback_start", endpoint="/playback-start")
            try:
                from .speech_out import notify_playback_started

                notify_playback_started(event_id)
                summarize_event(event_id)
            except Exception as exc:
                print(f"playback-start state update failed: {exc}")
            _json_response(self, 200, {"ok": True})
            return

        if parsed.path != "/command":
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"not found")
            return

        payload = _payload_from_query(parse_qs(parsed.query))
        if is_post:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(content_length) if content_length > 0 else b""
            body_payload = _payload_from_body(body)
            payload.update(body_payload)

        config = Config.from_env()
        event_id = _coerce_text(payload.get("eventId") or payload.get("event_id"))
        if event_id:
            _log_tts(event_id, "android_command_received", endpoint="/command")
        answer, intent, source = _assistant_answer_from_payload(payload, config)
        response_format = _coerce_text(payload.get("format") or payload.get("response_format") or "audio").lower()
        mode = _coerce_text(payload.get("mode") or payload.get("request_mode") or "assistant").lower()

        if response_format == "json":
            if not event_id:
                event_id = str(_now_ms())
            record = prepare_glasses_speech(answer, event_id)
            phrase_id = record.get("phraseId", "") if record else ""
            _json_response(
                self,
                200,
                {
                    "mode": mode,
                    "intent": intent,
                    "source": source,
                    "answer": answer,
                    "phraseId": phrase_id,
                    "eventId": event_id,
                    "audio_url": public_tts_url(event_id) if record else "",
                    "audioUrl": public_tts_url(event_id) if record else "",
                },
            )
            return

        self._send_wav(answer, event_id=event_id)

    def _send_wav(self, text: str, sample_rate: int = 16000, channels: int = 1, event_id: str = "", phrase_id: str = ""):
        download_started = time.monotonic()
        event_id = event_id or str(_now_ms())
        log_stage("AUDIO_DOWNLOAD_START", event_id=event_id)
        _log_tts(event_id, "audio_download_start")
        try:
            with _tts_cache_lock:
                cached = _tts_cache.get(event_id)
            if phrase_id:
                common = lookup_by_phrase_id(phrase_id)
                if not common:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b"common phraseId not found")
                    return
                cached = _record_from_common(common, event_id, _now_ms())
                with _tts_cache_lock:
                    _tts_cache[event_id] = cached
            if cached:
                wav = cached["wav"]
                sample_rate = int(cached["sampleRate"])
                channels = int(cached["channels"])
            else:
                if event_id and not text:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b"TTS eventId not found")
                    return
                record = prepare_glasses_speech(text, event_id, sample_rate=sample_rate, channels=channels)
                wav = record["wav"]
                cached = record
        except Exception as exc:
            message = f"TTS failed: {exc}".encode("utf-8", errors="replace")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(message)
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("X-Sample-Rate", str(sample_rate))
        self.send_header("X-Channels", str(channels))
        self.send_header("X-Bits-Per-Sample", "16")
        self.send_header("X-Event-Id", event_id)
        self.send_header("X-Phrase-Id", str(cached.get("phraseId", "")) if cached else "")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        for offset in range(0, len(wav), 16384):
            self.wfile.write(wav[offset : offset + 16384])
            self.wfile.flush()
        elapsed = int((time.monotonic() - download_started) * 1000)
        with _tts_cache_lock:
            cached = _tts_cache.get(event_id) or {}
        total = _now_ms() - int(cached.get("textReadyAtMs", cached.get("createdAtMs", _now_ms())))
        log_stage("AUDIO_DOWNLOAD_END", event_id=event_id, audioBytes=len(wav), downloadMs=elapsed)
        _log_tts(event_id, "audio_download_end", downloadMs=elapsed, audioBytes=len(wav), totalTextToAudioMs=total)
        log_stage("AUDIO_READY_FOR_ANDROID", event_id=event_id, inferred=True)
        _log_tts(event_id, "audio_ready_for_android", inferred="audio_download_complete")

    def log_message(self, format, *args):
        try:
            message = format % args
        except Exception:
            message = str(args)
        if "/pending" in message:
            return
        status = ""
        parts = message.rsplit(" ", 2)
        if parts:
            status = parts[-2] if parts[-1].isdigit() and len(parts) > 1 else parts[-1]
        if status.startswith(("4", "5")):
            print(f"backend-http: {message}")
        else:
            app_log.debug(f"backend-http: {message}")


class TtsHttpServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        try:
            self._server = ThreadingHTTPServer((host, port), _BackendHandler)
        except OSError as exc:
            raise SystemExit(f"MAC_BACKEND\nstatus=FAILED\nendpoint={host}:{port}\nerror={exc}") from exc
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        print(f"MAC_BACKEND\nstatus=LISTENING\nendpoint={self.host}:{self.port}")
        return self

    @property
    def address(self):
        return self._server.server_address

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
