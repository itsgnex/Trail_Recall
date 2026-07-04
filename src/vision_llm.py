import base64

import cv2

from .ollama_client import generate_json


def crop_to_base64(crop):
    h, w = crop.shape[:2]
    scale = 640 / max(h, w)
    if scale < 1:
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return base64.b64encode(encoded).decode() if ok else ""


def path_to_base64(image_path):
    image = cv2.imread(str(image_path))
    return crop_to_base64(image) if image is not None else ""


def analyze_image_with_gemma(crop, detected_kind, user_action, config, ocr_text="", clip_confidence=None):
    image = crop_to_base64(crop)
    if not image:
        return None
    model = getattr(config, "image_task_model", getattr(config, "vision_llm_model", "gemma3:4b"))
    print(f"sending crop to vision model {model}")
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
    data = generate_json(prompt, config, timeout=config.ollama_image_timeout, model=model, images=[image])
    if not data:
        return None
    print(f'vision llm: {data.get("image_type")}, visible_text="{data.get("visible_text")}", confidence={data.get("confidence")}')
    return data


def verify_same_sign(current_crop_path, previous_memory, config):
    if previous_memory is None:
        return {"same_sign": False, "reason": "no previous sign memory", "confidence": 0.0}

    current_image = path_to_base64(current_crop_path)
    previous_image = path_to_base64(previous_memory.get("crop_path"))
    model = getattr(config, "image_task_model", getattr(config, "vision_llm_model", "gemma3:4b"))
    print(f"sign memory: verifying duplicate with {model}")

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
        data = generate_json(prompt, config, timeout=getattr(config, "ollama_image_timeout", 180), model=model, images=images)
    else:
        data = generate_json(prompt, config, timeout=getattr(config, "ollama_text_timeout", 60), model=model)
    if not data:
        return {"same_sign": False, "reason": "no model result", "confidence": 0.0}
    return data
