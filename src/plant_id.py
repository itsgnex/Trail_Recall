from dataclasses import dataclass
import json
import urllib.parse
import urllib.request

import cv2
import numpy as np


@dataclass
class PlantIdResult:
    provider: str = "plantnet"
    success: bool = False
    common_name: str | None = None
    scientific_name: str | None = None
    family: str | None = None
    genus: str | None = None
    score: float = 0.0
    raw_top_result: dict | None = None
    error: str | None = None

    @property
    def confidence_level(self):
        if self.score >= 0.70:
            return "high"
        if self.score >= 0.45:
            return "medium"
        return "low"


_warned_unavailable = False


def _encode_crop(crop):
    h, w = crop.shape[:2]
    scale = 1024 / max(h, w)
    if scale < 1:
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return encoded.tobytes() if ok else None


def identify_plant(crop, config):
    global _warned_unavailable
    api_key = getattr(config, "plantnet_api_key", "")
    if not getattr(config, "use_plant_id", False) or getattr(config, "plant_id_provider", "plantnet") != "plantnet":
        if not _warned_unavailable:
            print("plant id: unavailable, using vision description fallback")
            _warned_unavailable = True
        return PlantIdResult(error="PlantNet disabled")
    if not api_key:
        if not _warned_unavailable:
            print("plant id: unavailable, using vision description fallback")
            _warned_unavailable = True
        return PlantIdResult(success=False, error="PlantNet API key missing")

    payload = _encode_crop(crop)
    if not payload:
        return PlantIdResult(success=False, error="Unable to encode image")

    try:
        print("plant id: sending crop to Pl@ntNet")
        project = getattr(config, "plantnet_project", "all")
        lang = getattr(config, "plantnet_lang", "en")
        organ = getattr(config, "plantnet_organ", "auto")
        query = urllib.parse.urlencode({"api-key": api_key, "lang": lang})
        url = f"https://my-api.plantnet.org/v2/identify/{project}?{query}"
        boundary = "----metra-live-plantnet"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="images"; filename="crop.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode() + payload + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="organs"\r\n\r\n{organ}\r\n'
            f"--{boundary}--\r\n"
        ).encode()
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode())

        result = (data.get("results") or [{}])[0]
        species = result.get("species") or {}
        common_names = species.get("commonNames") or []
        common_name = common_names[0] if common_names else None
        scientific_name = species.get("scientificNameWithoutAuthor")
        family = (species.get("family") or {}).get("scientificName")
        genus = (species.get("genus") or {}).get("scientificName")
        score = float(result.get("score", 0.0))
        print(f"plant id: top result common_name={common_name}, scientific_name={scientific_name}, score={score:.2f}")
        print(f"plant id: confidence {PlantIdResult(score=score).confidence_level}")
        return PlantIdResult(
            success=bool(common_name or scientific_name),
            common_name=common_name,
            scientific_name=scientific_name,
            family=family,
            genus=genus,
            score=score,
            raw_top_result=result,
        )
    except Exception as exc:
        print(f"plant id: unavailable ({exc}), using vision description fallback")
        return PlantIdResult(success=False, error=str(exc))


def describe_plant(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 35, 35), (90, 255, 255))
    green_ratio = float(np.count_nonzero(green)) / green.size
    if green_ratio > 0.18:
        return "This may be a plant, but I am not certain of the exact type. I can describe what I see or give more detail if you would like."
    return "I am not certain this is a plant from the camera view. Try holding it in the center with better light."
