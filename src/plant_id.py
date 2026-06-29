import cv2
import numpy as np
import json
import uuid
import urllib.request


_warned_unavailable = False


def identify_plant(crop, config):
    global _warned_unavailable
    if not getattr(config, "use_plant_id", False) or not getattr(config, "plantnet_api_key", ""):
        if not _warned_unavailable:
            print("plant id: unavailable, using vision description fallback")
            _warned_unavailable = True
        return None

    try:
        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            return None

        boundary = f"----metra-live-{uuid.uuid4().hex}"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="images"; filename="crop.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode() + encoded.tobytes() + f"\r\n--{boundary}--\r\n".encode()
        url = f"https://my-api.plantnet.org/v2/identify/all?api-key={config.plantnet_api_key}"
        request = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode())

        result = (data.get("results") or [{}])[0]
        species = result.get("species") or {}
        common_names = species.get("commonNames") or []
        name = common_names[0] if common_names else species.get("scientificNameWithoutAuthor")
        confidence = float(result.get("score", 0))
        print(f"plant id: {name or 'uncertain'}, confidence={confidence:.2f}")
        return {
            "common_name": name,
            "scientific_name": species.get("scientificNameWithoutAuthor"),
            "confidence": confidence,
        }
    except Exception as exc:
        print(f"plant id: unavailable ({exc}), using vision description fallback")
        return None


def describe_plant(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 35, 35), (90, 255, 255))
    green_ratio = float(np.count_nonzero(green)) / green.size
    if green_ratio > 0.18:
        return "This may be a plant, but I am not certain of the exact type. I can describe what I see or give more detail if you would like."
    return "I am not certain this is a plant from the camera view. Try holding it in the center with better light."
