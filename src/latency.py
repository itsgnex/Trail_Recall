from __future__ import annotations

import contextvars
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from . import app_log

_interaction_id = contextvars.ContextVar("interaction_id", default="")
_event_id = contextvars.ContextVar("event_id", default="")
_start = contextvars.ContextVar("latency_start", default=0.0)
_previous = contextvars.ContextVar("latency_previous", default=0.0)
_stages = contextvars.ContextVar("latency_stages", default=None)
_event_contexts = {}
_event_lock = threading.Lock()


def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _threshold_ms():
    try:
        return int(float(os.getenv("LATENCY_WARN_THRESHOLD_MS", "1000")))
    except ValueError:
        return 1000


def _jsonl_enabled():
    return os.getenv("LATENCY_LOG_FILE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _console_detail_enabled():
    return os.getenv("LATENCY_CONSOLE_MODE", "summary").strip().lower() in {"debug", "trace", "full"} or app_log.enabled("DEBUG")


def _warning_enabled_for_stage(stage: str) -> bool:
    expected_realtime = {
        "WAKE_AUDIO_CAPTURE_END",
        "COMMAND_CAPTURE_END",
        "FOLLOWUP_CAPTURE_END",
        "PLAYBACK_ESTIMATED_END",
        "BLUETOOTH_PLAYBACK_START",
        "TTS_STATE_CLEARED",
        "WAKE_LISTENER_RESUMED",
    }
    if stage in expected_realtime:
        return False
    return _console_detail_enabled() or stage.endswith("_ERROR") or stage.endswith("_TIMEOUT")


def new_interaction(interaction_id: str | None = None) -> str:
    value = interaction_id or uuid.uuid4().hex[:8]
    now = time.monotonic()
    _interaction_id.set(value)
    _event_id.set("")
    _start.set(now)
    _previous.set(now)
    _stages.set([])
    return value


def clear_interaction():
    _interaction_id.set("")
    _event_id.set("")
    _start.set(0.0)
    _previous.set(0.0)
    _stages.set([])


def set_interaction(interaction_id: str, started_at: float | None = None, stages: list | None = None):
    now = time.monotonic()
    _interaction_id.set(interaction_id)
    _start.set(started_at or now)
    stage_list = stages if stages is not None else []
    _previous.set(stage_list[-1]["monotonic"] if stage_list and "monotonic" in stage_list[-1] else started_at or now)
    _stages.set(stage_list)


def current_interaction_id() -> str:
    return _interaction_id.get()


def snapshot():
    return {
        "interactionId": _interaction_id.get(),
        "startedAt": _start.get(),
        "stages": _stages.get() if isinstance(_stages.get(), list) else [],
    }


def set_event(event_id: str):
    _event_id.set(event_id or "")
    if event_id:
        with _event_lock:
            _event_contexts[event_id] = snapshot()


def _restore_event_context(event_id: str | None):
    if not event_id or current_interaction_id():
        return
    with _event_lock:
        ctx = _event_contexts.get(event_id)
    if ctx:
        set_interaction(ctx.get("interactionId") or "startup", started_at=ctx.get("startedAt"), stages=ctx.get("stages"))


def log_stage(stage: str, event_id: str | None = None, **values):
    if event_id is not None:
        _restore_event_context(event_id)
    now = time.monotonic()
    started = _start.get() or now
    previous = _previous.get() or started
    elapsed = int((now - previous) * 1000)
    total = int((now - started) * 1000)
    _previous.set(now)
    item = {
        "timestamp": _now_iso(),
        "interactionId": _interaction_id.get() or "startup",
        "eventId": event_id if event_id is not None else _event_id.get(),
        "stage": stage,
        "elapsedMs": elapsed,
        "totalMs": total,
        "thread": threading.current_thread().name,
        "monotonic": now,
    }
    item.update({key: value for key, value in values.items() if key.lower() not in {"api_key", "authorization"}})
    stages = _stages.get()
    if isinstance(stages, list):
        stages.append(item)
    if _console_detail_enabled():
        print("LATENCY")
        for key in ("timestamp", "interactionId", "eventId", "stage", "elapsedMs", "totalMs", "thread"):
            if item.get(key) != "":
                print(f"{key}={item[key]}")
        for key, value in item.items():
            if key not in {"timestamp", "interactionId", "eventId", "stage", "elapsedMs", "totalMs", "thread", "monotonic"}:
                print(f"{key}={value}")
    if elapsed > _threshold_ms() and _warning_enabled_for_stage(stage):
        print("LATENCY_WARNING")
        print(f"interactionId={item['interactionId']}")
        print(f"stage={stage}")
        print(f"elapsedMs={elapsed}")
        print(f"thresholdMs={_threshold_ms()}")
    if _jsonl_enabled():
        path = Path("logs") / "latency.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return item


def summarize(interaction_id: str | None = None):
    stages = list(_stages.get() or [])
    if not stages:
        return {}
    wanted = {
        "wakeCaptureMs": ("WAKE_LISTEN_START", "WAKE_AUDIO_CAPTURE_END"),
        "wakeTranscriptionMs": ("WAKE_TRANSCRIPTION_START", "WAKE_TRANSCRIPTION_END"),
        "commandCaptureMs": ("COMMAND_CAPTURE_START", "COMMAND_CAPTURE_END"),
        "commandTranscriptionMs": ("OPENROUTER_STT_START", "TRANSCRIPT_READY"),
        "intentRoutingMs": ("INTENT_ROUTING_START", "INTENT_ROUTING_END"),
        "visionMs": ("VISION_REQUEST_START", "VISION_REQUEST_END"),
        "commonAudioLookupMs": ("COMMON_AUDIO_LOOKUP_START", "COMMON_AUDIO_HIT"),
        "ttsGenerationMs": ("TTS_GENERATION_START", "TTS_GENERATION_END"),
        "androidHandoffMs": ("ANDROID_SPEAK_REQUEST_START", "ANDROID_SPEAK_REQUEST_END"),
        "playbackStartMs": ("RESPONSE_TEXT_READY", "BLUETOOTH_PLAYBACK_START"),
    }
    by_stage = {}
    for item in stages:
        by_stage.setdefault(item["stage"], item)

    summary = {"interactionId": interaction_id or current_interaction_id() or stages[0]["interactionId"]}
    for name, (start_stage, end_stage) in wanted.items():
        start = by_stage.get(start_stage)
        end = by_stage.get(end_stage)
        summary[name] = max(0, end["totalMs"] - start["totalMs"]) if start and end else ""
    summary["totalTimeToFirstAudioMs"] = by_stage.get("BLUETOOTH_PLAYBACK_START", by_stage.get("PLAYBACK_ESTIMATED_END", stages[-1]))["totalMs"]
    summary["totalInteractionMs"] = stages[-1]["totalMs"]
    slowest = max(stages, key=lambda item: item.get("elapsedMs", 0))
    summary["slowestStage"] = slowest["stage"]
    summary["slowestStageMs"] = slowest["elapsedMs"]

    print(
        "LATENCY_SUMMARY "
        f"interactionId={summary['interactionId']} "
        f"wakeCaptureMs={summary['wakeCaptureMs']} "
        f"wakeSttMs={summary['wakeTranscriptionMs']} "
        f"commandCaptureMs={summary['commandCaptureMs']} "
        f"commandSttMs={summary['commandTranscriptionMs']} "
        f"ttsMs={summary['ttsGenerationMs']} "
        f"androidMs={summary['androidHandoffMs']} "
        f"firstAudioMs={summary['totalTimeToFirstAudioMs']} "
        f"slowest={summary['slowestStage']}:{summary['slowestStageMs']}"
    )
    return summary


def summarize_event(event_id: str):
    _restore_event_context(event_id)
    return summarize()
