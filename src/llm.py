from .intent import Intent
from .ollama_client import generate
from .ocr import read_text
from .plant_id import PlantIdResult, describe_plant, identify_plant
from .phrases import (
    get_clarification_response,
    get_repeat_response,
    get_unclear_plant_response,
    get_unclear_sign_response,
)
from .vision_llm import analyze_image_with_gemma


RULES = (
    "You are an assistive smart-glasses voice assistant for older adults. "
    "Speak respectfully and naturally. Do not use baby talk or patronizing language. "
    "Never say dear, sweetie, grandpa, or similar terms. "
    "Do not use emotional filler like lovely or wonderful. "
    "Keep the first answer short, clear, and useful, about one or two sentences. "
    "If uncertain, say so. Use plain language. Give one useful next option only when helpful."
)


def known_sign_meaning(*parts):
    text = " ".join(str(part or "").lower() for part in parts)
    mappings = [
        (("walking person", "person walking", "person crossing", "pedestrian"), "This looks like a pedestrian crossing sign. It means people may be walking or crossing nearby, so please watch carefully."),
        (("stop", "red octagon"), "This is a stop sign. It means you should stop and check before moving ahead."),
        (("one way",), "This is a one-way sign. It means movement or traffic should go only in the direction shown."),
        (("left arrow", "arrow left"), "This sign points left. It likely means you should follow or pay attention to the direction shown."),
        (("right arrow", "arrow right"), "This sign points right. It likely means you should follow or pay attention to the direction shown."),
        (("merge", "lanes merge", "two lanes merge"), "This looks like a lane merge warning sign. It means lanes join ahead, so be ready to merge carefully."),
        (("no entry", "do not enter"), "This looks like a do-not-enter sign. It means you should not go that way."),
        (("exit",), "This looks like an exit sign. It points toward a way out."),
        (("caution", "warning triangle", "warning"), "This looks like a warning sign. It means you should use caution nearby."),
        (("wheelchair", "accessible"), "The symbol indicates an accessible route or facility."),
        (("restroom", "washroom", "toilet"), "The sign indicates a restroom nearby."),
    ]
    for keys, meaning in mappings:
        if any(key in text for key in keys):
            return meaning
    return ""


def naturalize_sign_response(vision_result):
    if not vision_result:
        return None

    visible_text = (vision_result.get("visible_text") or "").strip()
    symbol = (vision_result.get("symbol_or_icon") or "").strip()
    description = (vision_result.get("description") or "").strip()
    plain_meaning = (vision_result.get("plain_meaning") or "").strip()
    is_clear = bool(vision_result.get("is_clear_enough", True))
    confidence = float(vision_result.get("confidence") or 0)

    if not is_clear or confidence < 0.35:
        return None

    meaning = plain_meaning or known_sign_meaning(visible_text, symbol, description)
    if not meaning:
        return None

    if visible_text and "stop" in visible_text.lower():
        lead = "This is a stop sign."
    elif visible_text and "one way" in visible_text.lower():
        lead = "This is a one-way sign."
    elif "pedestrian" in meaning.lower():
        lead = "This looks like a pedestrian crossing sign."
    elif "lane merge" in meaning.lower():
        lead = "This looks like a lane merge warning sign."
    elif "do-not-enter" in meaning.lower():
        lead = "This looks like a do-not-enter sign."
    elif "exit" in meaning.lower():
        lead = "This looks like an exit sign."
    elif symbol:
        lead = f"This sign points to {symbol}."
    else:
        lead = "This looks like a sign."

    cleaned = meaning.strip()
    for prefix in ("Warning:", "Description:", "Meaning:", "Recommended action:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    if cleaned.lower().startswith("it means "):
        cleaned = cleaned[9:]
    if cleaned.lower().startswith("it likely means "):
        cleaned = cleaned[16:]
    if lead.lower() in cleaned.lower():
        return cleaned
    return f"{lead} It means {cleaned}"


def answer_for(kind, crop, intent, config=None):
    fallback = fallback_answer(kind, crop, intent)
    if not getattr(config, "use_ollama", False):
        return fallback

    ocr_text = read_text(crop) if kind == "sign" else ""
    image_analysis = analyze_image_with_gemma(crop, kind, intent.value, config, ocr_text)
    plant_id = identify_plant(crop, config) if kind == "plant" else None

    if kind == "sign":
        visible_text = (image_analysis or {}).get("visible_text") or ""
        symbol = (image_analysis or {}).get("symbol_or_icon") or ""
        plain_meaning = (image_analysis or {}).get("plain_meaning") or ""
        recommended_action = (image_analysis or {}).get("recommended_action") or ""
        description = (image_analysis or {}).get("description") or ""
        is_clear = bool((image_analysis or {}).get("is_clear_enough"))
        confidence = float((image_analysis or {}).get("confidence") or 0)
        if not image_analysis:
            return fallback
        if not visible_text and not symbol and (not is_clear or confidence < 0.35):
            return get_unclear_sign_response()
        mapped_meaning = known_sign_meaning(visible_text, symbol, plain_meaning, description)
        natural = naturalize_sign_response(image_analysis)
        if natural:
            return natural
        if plain_meaning:
            return plain_meaning
        if visible_text and mapped_meaning:
            return mapped_meaning
        if mapped_meaning:
            return mapped_meaning
        if symbol:
            return "This looks like a sign, but I cannot interpret it clearly. Please hold it straighter or move a little closer."
        if intent == Intent.READ_SIGN_TEXT:
            task = "Read the sign text clearly. Do not add extra explanation unless needed."
        else:
            task = "Explain the sign or symbol meaning in plain language."
        prompt = f"""
{RULES}
Do not start with labels like Warning, Description, Meaning, or Recommended action. Speak as a natural assistant. Use one or two short sentences. Avoid repeating the same idea.
Task: {task}
Detected kind: {kind}
Dialogue action: {intent.value}
Gemma image analysis JSON: {image_analysis}
OCR text, if clean enough to support the answer: {ocr_text!r}
If visible_text is clear, say: The sign says: '...'. It means ...
If there is no readable text but a clear symbol or icon, explain the symbol cautiously.
If unclear, say: I can see what may be a sign, but I cannot read or interpret it clearly. Please hold it straighter or move a little closer.
Do not explain noisy OCR text.
"""
    else:
        if isinstance(plant_id, PlantIdResult) and not plant_id.success:
            plant_id_payload = {
                "provider": plant_id.provider,
                "success": plant_id.success,
                "common_name": plant_id.common_name,
                "scientific_name": plant_id.scientific_name,
                "family": plant_id.family,
                "genus": plant_id.genus,
                "score": plant_id.score,
                "confidence_level": plant_id.confidence_level,
                "raw_top_result": plant_id.raw_top_result,
                "error": plant_id.error,
            }
        elif isinstance(plant_id, PlantIdResult):
            plant_id_payload = {
                "provider": plant_id.provider,
                "success": plant_id.success,
                "common_name": plant_id.common_name,
                "scientific_name": plant_id.scientific_name,
                "family": plant_id.family,
                "genus": plant_id.genus,
                "score": plant_id.score,
                "confidence_level": plant_id.confidence_level,
                "raw_top_result": plant_id.raw_top_result,
                "error": plant_id.error,
            }
        else:
            plant_id_payload = plant_id
        vision_description = (image_analysis or {}).get("description") or ""
        prompt = f"""
{RULES}
Speak naturally and avoid filler. Do not repeat the same idea.
Detected kind: {kind}
Dialogue action: {intent.value}
Gemma image analysis JSON: {image_analysis}
Plant ID result: {plant_id_payload}
Vision description: {vision_description}
If plant ID has a high-confidence common name, say: This appears to be [common name]. It is [short plain-language description].
If plant ID has medium confidence, say it may be [common name], but you are not fully certain.
If plant ID is low confidence or unavailable, say the exact type is uncertain and describe what is visible.
Do not invent a species.
"""
    model = getattr(config, "final_response_model", "gemma3:4b")
    print(f"final response model: {model}")
    return generate(prompt, config, timeout=getattr(config, "ollama_text_timeout", 60), model=model) or fallback


def fallback_answer(kind, crop, intent):
    if kind == "sign":
        text = read_text(crop)
        if text:
            if intent == Intent.READ_SIGN_TEXT:
                return f"The sign appears to say: {text}. Would you like me to explain it?"
            return f"This appears to be a sign with the text: {text}. Would you like more detail?"
        return get_unclear_sign_response()

    return describe_plant(crop) or get_unclear_plant_response()


def _demo():
    assert "pedestrian crossing" in known_sign_meaning("yellow diamond", "walking person symbol")
    assert "pedestrian crossing" in known_sign_meaning("yellow diamond shape with a black silhouette of a person walking")
    assert "stop" in known_sign_meaning("STOP", "red octagon").lower()


if __name__ == "__main__":
    _demo()
