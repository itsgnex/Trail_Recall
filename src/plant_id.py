from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import requests


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
            print("plant id: unavailable, API key missing")
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
        url = f"https://my-api.plantnet.org/v2/identify/{project}?api-key={api_key}"
        print(f"plant id: endpoint=https://my-api.plantnet.org/v2/identify/{project}?api-key=REDACTED")
        params = {"lang": lang, "nb-results": 3, "no-reject": "true"}
        files = {"images": ("crop.jpg", payload, "image/jpeg")}
        data = {"organs": organ} if organ and organ != "auto" else None
        response = requests.post(url, params=params, files=files, data=data, timeout=30)
        if response.status_code == 401 or response.status_code == 403:
            print("plant id: unauthorized, check PlantNet API key")
            return PlantIdResult(success=False, error=f"PlantNet unauthorized ({response.status_code})")
        if response.status_code == 404:
            print("plant id: no plant match returned")
            return PlantIdResult(success=False, error="No plant match returned")
        if response.status_code == 429:
            print("plant id: rate limit reached")
            return PlantIdResult(success=False, error="PlantNet rate limit reached")
        response.raise_for_status()
        data = response.json()

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
        return "This may be a plant, but I am not certain of the exact type."
    return "I am not certain this is a plant from the camera view. Try holding it in the center with better light."


if __name__ == "__main__":
    from .config import Config

    if len(sys.argv) != 2:
        print("usage: python -m src.plant_id path/to/image.jpg")
        raise SystemExit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"file not found: {image_path}")
        raise SystemExit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"unable to read image: {image_path}")
        raise SystemExit(1)

    result = identify_plant(image, Config.from_env())
    print(json.dumps(result.__dict__, indent=2))
