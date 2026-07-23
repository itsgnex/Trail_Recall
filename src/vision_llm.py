import base64
import hashlib
import json

import cv2

from .inference import local_model as local_inference_model, visual_inference_allowed
from .ollama_client import generate_json
from .openrouter_client import generate as generate_openrouter

MAX_IMAGE_EDGE = 384


def crop_to_base64(crop):
    h, w = crop.shape[:2]
    scale = MAX_IMAGE_EDGE / max(h, w)
    if scale < 1:
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    return base64.b64encode(encoded).decode() if ok else ""


def _vision_timeout(config):
    return int(getattr(config, "openrouter_timeout", 8))


def _parse_json(text):
    if not text:
        return None
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


def path_to_base64(image_path):
    image = cv2.imread(str(image_path))
    return crop_to_base64(image) if image is not None else ""


def crop_signature(crop):
    if crop is None or crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    return hashlib.sha1(small.tobytes()).hexdigest()


def _openrouter_json(prompt, config, images=None, label="vision"):
    openrouter_model = getattr(config, "openrouter_model", "google/gemini-3.1-flash-lite")
    if images:
        print(f"{label}: sending image to OpenRouter model {openrouter_model}")
    else:
        print(f"{label}: sending text to OpenRouter model {openrouter_model}")
    text = generate_openrouter(prompt, config, timeout=_vision_timeout(config), model=openrouter_model, images=images)
    return _parse_json(text)


def _local_json(prompt, config, images=None):
    if not getattr(config, "use_local_vision_llm", False):
        return None
    if not visual_inference_allowed():
        return None
    local_model = getattr(config, "image_task_model", getattr(config, "vision_llm_model", "gemma3:1b"))
    kind = "image" if images else "text"
    print(f"local vision fallback: calling {local_model} for {kind}")
    timeout = (
        getattr(config, "ollama_image_timeout", 45) if images else getattr(config, "ollama_text_timeout", 20)
    )
    with local_inference_model("visual"):
        return generate_json(prompt, config, timeout=timeout, model=local_model, images=images)


def analyze_image_with_gemma(crop, detected_kind, user_action, config, ocr_text="", clip_confidence=None):
    image = crop_to_base64(crop)
    if not image:
        return None
    prompt = f"""
Look only at the provided image crop.
Do not invent details.
If it is a sign, explain what the sign likely means, not just what it looks like.
A sign can contain text, symbols, icons, arrows, warning triangles, colors, or pictograms.
Do not require text to classify something as a sign.
If readable text is visible, read only clear text. If text is unclear, set visible_text to null.
If it is a symbol-only sign, identify the likely meaning of the symbol.
If a warning/caution icon is visible, describe the symbol and likely caution meaning.
If the image is blurry, angled, too small, or unclear, set is_clear_enough to false.
If it is a plant, do not guess exact species unless clearly identifiable.
If uncertain, set uncertain to true.
Do not invent text.
Do not over-explain.
Return strict JSON only:
{{"image_type":"plant|sign|symbol_sign|text_sign|other|uncertain","visible_text":null,"symbol_or_icon":null,"description":"","plain_meaning":null,"recommended_action":null,"is_clear_enough":false,"confidence":0.0,"uncertain":true}}

Example 1:
Image: yellow diamond sign with black walking person
Return: {{"image_type":"symbol_sign","visible_text":null,"symbol_or_icon":"walking person symbol","description":"A yellow diamond warning sign with a walking person symbol.","plain_meaning":"Pedestrians may be crossing or walking nearby.","recommended_action":"Watch carefully for people walking or crossing.","is_clear_enough":true,"confidence":0.9,"uncertain":false}}

Example 2:
Image: red octagon with STOP
Return: {{"image_type":"text_sign","visible_text":"STOP","symbol_or_icon":"red octagon","description":"A red stop sign.","plain_meaning":"This appears to be a stop sign. Please stop here and check carefully before continuing.","recommended_action":"Please stop here and check carefully before continuing.","is_clear_enough":true,"confidence":0.95,"uncertain":false}}

Example 3:
Image: black arrow on sign
Return: {{"image_type":"symbol_sign","visible_text":null,"symbol_or_icon":"direction arrow","description":"A sign with an arrow.","plain_meaning":"The arrow shows the direction to follow.","recommended_action":"Follow the direction of the arrow if it applies to your path.","is_clear_enough":true,"confidence":0.85,"uncertain":false}}

Context:
- detected_kind: {detected_kind}
- user_action: {user_action}
- ocr_text: {(ocr_text or None)!r}
- clip_confidence: {clip_confidence}
"""
    data = _openrouter_json(prompt, config, images=[image], label="vision")
    if not data:
        data = _local_json(prompt, config, images=[image])
    if not data:
        return None
    print(f'vision llm: {data.get("image_type")}, visible_text="{data.get("visible_text")}", confidence={data.get("confidence")}')
    return data


def verify_plant_candidate(crop, config):
    image = crop_to_base64(crop)
    if not image:
        return {"is_plant": False, "confidence": 0.0, "object_type": "unknown", "reason": "no image"}

    prompt = """
Look only at the provided camera crop.
Decide whether the main centered object is a real living plant or a clearly visible plant part such as leaves, stems, flowers, or a potted plant.
Reject water bottles, cups, plastic objects, posters, printed pictures, screens, clothing, signs, people, furniture, and random green or blue objects.
Do not treat a plant picture, plant label, or decorative pattern as a real plant.
If unsure, return is_plant false.
Return strict JSON only:
{"is_plant": false, "object_type": "water bottle", "reason": "The centered object is a bottle, not plant leaves or flowers.", "confidence": 0.95}
"""
    data = _openrouter_json(prompt, config, images=[image], label="plant verify")
    if not data:
        data = _local_json(prompt, config, images=[image])
    if not data:
        return {"is_plant": False, "confidence": 0.0, "object_type": "unknown", "reason": "no verification result"}
    print(f'plant verify: is_plant={data.get("is_plant")}, object_type="{data.get("object_type")}", confidence={data.get("confidence")}')
    return data


def verify_same_sign(current_crop_path, previous_memory, config):
    if previous_memory is None:
        return {"same_sign": False, "reason": "no previous sign memory", "confidence": 0.0}

    current_image = path_to_base64(current_crop_path)
    previous_image = path_to_base64(previous_memory.get("crop_path"))
    print("sign memory: verifying duplicate")

    prompt = f"""
You are comparing a current sign image against a previously seen sign. Decide if they are the same physical sign or effectively the same sign content. Return strict JSON only.
If uncertain, return same_sign=false.
Previous sign memory:
- visible_text: {(previous_memory.get("visible_text") or None)!r}
- symbol_or_icon: {(previous_memory.get("symbol_or_icon") or None)!r}
- plain_meaning: {(previous_memory.get("plain_meaning") or None)!r}
- description: {(previous_memory.get("description") or None)!r}
- final_answer: {(previous_memory.get("final_answer") or None)!r}
Return:
{{"same_sign": true, "reason": "Both appear to be the same sign.", "confidence": 0.92}}
    """
    images = [image for image in (current_image, previous_image) if image]
    if images:
        data = _openrouter_json(prompt, config, images=images, label="sign memory")
        if not data:
            data = _local_json(prompt, config, images=images)
    else:
        data = _openrouter_json(prompt, config, label="sign memory")
        if not data:
            data = _local_json(prompt, config)
    if not data:
        return {"same_sign": False, "reason": "no model result", "confidence": 0.0}
    return data
