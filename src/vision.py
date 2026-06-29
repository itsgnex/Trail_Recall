from dataclasses import dataclass

import cv2
import numpy as np

from .clip_classifier import classify_crop_with_clip
from .ocr import read_text


@dataclass(frozen=True)
class VisionResult:
    kind: str
    reason: str


def focus_crop(frame, fraction):
    h, w = frame.shape[:2]
    size = int(min(h, w) * fraction)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    return frame[y1 : y1 + size, x1 : x1 + size], (x1, y1, x1 + size, y1 + size)


def draw_focus_box(frame, box):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, "Focus area", (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def fallback_analyze_crop(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 35, 35), (90, 255, 255))
    green_ratio = float(np.count_nonzero(green)) / green.size
    if green_ratio > 0.18:
        return VisionResult("plant", f"fallback green ratio {green_ratio:.2f}")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size
    if edge_ratio > 0.09:
        return VisionResult("sign", f"fallback text-like edges {edge_ratio:.2f}")

    return VisionResult("other", "no plant or sign cue")


def analyze_crop(crop, config=None):
    text = read_text(crop)
    if len(text) >= 4:
        return VisionResult("sign", f"ocr text detected: {text[:40]}")

    if getattr(config, "use_clip", True):
        clip_result = classify_crop_with_clip(crop)
        if clip_result:
            return VisionResult(*clip_result)

    return fallback_analyze_crop(crop)
