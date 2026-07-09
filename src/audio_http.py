from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .config import Config
from .intent import Intent, classify_intent_with_source
from .llm import answer_for, answer_general_question_with_1b, describe_current_object
from .phrases import get_cancel_response, get_clarification_response, get_repeat_response
from .session_state import SessionState

_pending_speech: deque[str] = deque(maxlen=32)
_pending_trail: deque[str] = deque(maxlen=16)
_pending_lock = threading.Lock()


def enqueue_glasses_speech(text: str) -> None:
    cleaned = (text or "").strip()
    if not cleaned:
        return
    with _pending_lock:
        _pending_speech.append(cleaned)


def dequeue_glasses_speech() -> str:
    with _pending_lock:
        if _pending_speech:
            return _pending_speech.popleft()
    return ""


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
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if parsed.path == "/pending":
            text = dequeue_glasses_speech()
            _json_response(self, 200, {"text": text})
            return

        if parsed.path == "/pending-trail":
            action = dequeue_trail_command()
            _json_response(self, 200, {"action": action})
            return

        if parsed.path == "/tts":
            query = parse_qs(parsed.query)
            text = _coerce_text(query.get("text", [""])[0])
            rate = int(_coerce_text(query.get("rate", ["16000"])[0]) or "16000")
            channels = int(_coerce_text(query.get("channels", ["1"])[0]) or "1")
            self._send_wav(text, rate, channels)
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
        answer, intent, source = _assistant_answer_from_payload(payload, config)
        response_format = _coerce_text(payload.get("format") or payload.get("response_format") or "audio").lower()
        mode = _coerce_text(payload.get("mode") or payload.get("request_mode") or "assistant").lower()

        if response_format == "json":
            _json_response(
                self,
                200,
                {
                    "mode": mode,
                    "intent": intent,
                    "source": source,
                    "answer": answer,
                    "audio_url": f"/command?mode=tts&format=audio&text={answer}",
                },
            )
            return

        self._send_wav(answer)

    def _send_wav(self, text: str, sample_rate: int = 16000, channels: int = 1):
        try:
            wav = synthesize_wav(text, sample_rate=sample_rate, channels=channels)
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
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)

    def log_message(self, format, *args):
        try:
            message = format % args
        except Exception:
            message = str(args)
        if "/pending" in message:
            return
        print(f"backend-http: {message}")


class TtsHttpServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._server = ThreadingHTTPServer((host, port), _BackendHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        return self

    @property
    def address(self):
        return self._server.server_address

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
