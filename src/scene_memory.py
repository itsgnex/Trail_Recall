from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .vision_llm import crop_signature
from .plant_id import PlantIdResult

_LOCK = threading.RLock()
_MEMORY_PATH = Path(".scene_memory.json")
_MEMORIES: list[dict] = []
_LOADED = False


def _normalize_intent(intent):
    return getattr(intent, "value", intent) or ""


def _normalize_signature(crop):
    return crop_signature(crop) if crop is not None else ""


def _entry_key(kind, intent, signature):
    return f"{kind}|{_normalize_intent(intent)}|{signature}"


def _load():
    global _LOADED, _MEMORIES
    if _LOADED:
        return
    _LOADED = True
    if not _MEMORY_PATH.exists():
        _MEMORIES = []
        return
    try:
        data = json.loads(_MEMORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            _MEMORIES = [item for item in data if isinstance(item, dict)]
        else:
            _MEMORIES = []
    except Exception:
        _MEMORIES = []


def _save():
    tmp_path = _MEMORY_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(_MEMORIES, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(_MEMORY_PATH)


def _cleanup(now=None, ttl_seconds=3600):
    now = time.time() if now is None else now
    _load()
    with _LOCK:
        _MEMORIES[:] = [
            item
            for item in _MEMORIES
            if now - float(item.get("last_seen_time", item.get("first_seen_time", now))) <= ttl_seconds
        ]


def _plant_payload(payload):
    if isinstance(payload, dict):
        return payload
    return None


def restore_plant_id(payload):
    if not isinstance(payload, dict):
        return None
    try:
        return PlantIdResult(
            provider=payload.get("provider", "plantnet"),
            success=bool(payload.get("success", False)),
            common_name=payload.get("common_name"),
            scientific_name=payload.get("scientific_name"),
            family=payload.get("family"),
            genus=payload.get("genus"),
            score=float(payload.get("score", 0.0) or 0.0),
            raw_top_result=payload.get("raw_top_result"),
            error=payload.get("error"),
        )
    except Exception:
        return None


def lookup_scene_memory(kind, intent, crop, ttl_seconds=3600):
    signature = _normalize_signature(crop)
    if not signature:
        return None
    now = time.time()
    _cleanup(now, ttl_seconds)
    key = _entry_key(kind, intent, signature)
    with _LOCK:
        for item in reversed(_MEMORIES):
            if item.get("key") != key:
                continue
            if now - float(item.get("last_seen_time", item.get("first_seen_time", now))) > ttl_seconds:
                return None
            item["last_seen_time"] = now
            item["seen_count"] = int(item.get("seen_count", 1)) + 1
            _save()
            return item
    return None


def store_scene_memory(
    kind,
    intent,
    crop,
    answer,
    *,
    crop_path=None,
    vision_result=None,
    plant_id_result=None,
    ttl_seconds=3600,
    max_items=120,
):
    signature = _normalize_signature(crop)
    if not signature or not answer:
        return None

    now = time.time()
    _cleanup(now, ttl_seconds)
    key = _entry_key(kind, intent, signature)
    plant_payload = _plant_payload(plant_id_result)
    with _LOCK:
        for item in _MEMORIES:
            if item.get("key") != key:
                continue
            item["answer"] = answer
            item["crop_path"] = crop_path or item.get("crop_path")
            item["vision_result"] = vision_result if vision_result is not None else item.get("vision_result")
            item["plant_id_result"] = plant_payload if plant_payload is not None else item.get("plant_id_result")
            item["last_seen_time"] = now
            item["seen_count"] = int(item.get("seen_count", 1)) + 1
            _MEMORIES.sort(key=lambda entry: float(entry.get("last_seen_time", 0.0)), reverse=True)
            del _MEMORIES[max_items:]
            _save()
            return item

        item = {
            "key": key,
            "kind": kind,
            "intent": _normalize_intent(intent),
            "signature": signature,
            "crop_path": crop_path,
            "answer": answer,
            "vision_result": vision_result,
            "plant_id_result": plant_payload,
            "first_seen_time": now,
            "last_seen_time": now,
            "seen_count": 1,
        }
        _MEMORIES.append(item)
        _MEMORIES.sort(key=lambda entry: float(entry.get("last_seen_time", 0.0)), reverse=True)
        del _MEMORIES[max_items:]
        _save()
        return item
