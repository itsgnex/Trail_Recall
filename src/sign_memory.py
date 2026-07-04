from __future__ import annotations

import threading
import time
import uuid

import numpy as np

_SIGN_MEMORIES: list[dict] = []
_LOCK = threading.RLock()


def _vector(embedding):
    if embedding is None:
        return None
    array = np.asarray(embedding, dtype=float)
    if array.size == 0:
        return None
    return array


def _cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if not a_norm or not b_norm:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def cleanup_old_signs(now=None, ttl_seconds=600):
    now = time.monotonic() if now is None else now
    with _LOCK:
        _SIGN_MEMORIES[:] = [
            item
            for item in _SIGN_MEMORIES
            if now - float(item.get("last_seen_time", item.get("first_seen_time", now))) <= ttl_seconds
        ]


def find_similar_sign(embedding, min_similarity=0.82, now=None, ttl_seconds=600):
    now = time.monotonic() if now is None else now
    vector = _vector(embedding)
    if vector is None:
        return None, 0.0
    cleanup_old_signs(now, ttl_seconds)
    with _LOCK:
        best_item = None
        best_score = 0.0
        for item in _SIGN_MEMORIES:
            score = _cosine_similarity(vector, _vector(item.get("embedding")))
            if score > best_score:
                best_item = item
                best_score = score
        if best_item is None or best_score < min_similarity:
            return None, best_score
        return best_item, best_score


def is_recent_duplicate(memory_item, now=None, suppress_seconds=180):
    if not memory_item:
        return False
    now = time.monotonic() if now is None else now
    last_time = max(
        float(memory_item.get("last_prompted_time", 0.0)),
        float(memory_item.get("last_explained_time", 0.0)),
        float(memory_item.get("last_seen_time", 0.0)),
    )
    return last_time and (now - last_time) <= suppress_seconds


def add_or_update_sign_memory(
    *,
    crop_path=None,
    embedding=None,
    first_seen_time=None,
    last_seen_time=None,
    seen_count=1,
    last_prompted_time=None,
    last_explained_time=None,
    visible_text="",
    symbol_or_icon="",
    plain_meaning="",
    description="",
    final_answer="",
    max_items=30,
    ttl_seconds=600,
    duplicate_threshold=0.82,
    now=None,
):
    now = time.monotonic() if now is None else now
    cleanup_old_signs(now, ttl_seconds)
    vector = _vector(embedding)
    if vector is None:
        return None

    with _LOCK:
        match, score = find_similar_sign(vector, min_similarity=duplicate_threshold, now=now, ttl_seconds=ttl_seconds)
        if match is not None:
            match["embedding"] = vector.tolist()
            match["crop_path"] = crop_path or match.get("crop_path")
            match["last_seen_time"] = last_seen_time or now
            match["seen_count"] = int(match.get("seen_count", 1)) + 1
            if first_seen_time is not None:
                match["first_seen_time"] = first_seen_time
            if last_prompted_time is not None:
                match["last_prompted_time"] = last_prompted_time
            if last_explained_time is not None:
                match["last_explained_time"] = last_explained_time
            if visible_text:
                match["visible_text"] = visible_text
            if symbol_or_icon:
                match["symbol_or_icon"] = symbol_or_icon
            if plain_meaning:
                match["plain_meaning"] = plain_meaning
            if description:
                match["description"] = description
            if final_answer:
                match["final_answer"] = final_answer
            return match

        item = {
            "id": uuid.uuid4().hex,
            "crop_path": crop_path,
            "embedding": vector.tolist(),
            "first_seen_time": first_seen_time or now,
            "last_seen_time": last_seen_time or now,
            "seen_count": int(seen_count),
            "last_prompted_time": last_prompted_time or 0.0,
            "last_explained_time": last_explained_time or 0.0,
            "visible_text": visible_text,
            "symbol_or_icon": symbol_or_icon,
            "plain_meaning": plain_meaning,
            "description": description,
            "final_answer": final_answer,
        }
        _SIGN_MEMORIES.append(item)
        _SIGN_MEMORIES.sort(key=lambda entry: float(entry.get("last_seen_time", 0.0)), reverse=True)
        del _SIGN_MEMORIES[max_items:]
        return item


def _demo():
    a = [1.0, 0.0, 0.0]
    b = [0.98, 0.02, 0.0]
    c = [0.0, 1.0, 0.0]
    item = add_or_update_sign_memory(embedding=a, max_items=3)
    assert item is not None
    match, score = find_similar_sign(b)
    assert match is not None and score > 0.9
    match2, score2 = find_similar_sign(c)
    assert match2 is None or score2 < 0.82


if __name__ == "__main__":
    _demo()
