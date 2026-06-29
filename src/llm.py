from .intent import Intent
from .ollama_client import generate
from .ocr import read_text
from .plant_id import describe_plant, identify_plant
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
        (("walking person", "person walking", "person crossing", "pedestrian"), "This looks like a pedestrian crossing warning sign. It means people may be walking or crossing nearby, so please watch carefully."),
        (("stop", "red octagon"), "The sign means you should stop before continuing."),
        (("one way", "arrow", "direction"), "The sign shows the direction to follow."),
        (("no entry", "do not enter"), "The sign means you should not enter that area."),
        (("exit",), "The sign shows the direction to leave."),
        (("caution", "warning triangle", "warning"), "This looks like a warning sign. It means you should use caution nearby."),
        (("wheelchair", "accessible"), "The symbol indicates an accessible route or facility."),
        (("restroom", "washroom", "toilet"), "The sign indicates a restroom nearby."),
    ]
    for keys, meaning in mappings:
        if any(key in text for key in keys):
            return meaning
    return ""


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
            return "I can see what may be a sign, but I cannot read or interpret it clearly. Please hold it straighter or move a little closer."
        mapped_meaning = known_sign_meaning(visible_text, symbol, plain_meaning, description)
        if plain_meaning:
            if recommended_action and recommended_action.lower() not in plain_meaning.lower():
                return f"{plain_meaning} {recommended_action}"
            return plain_meaning
        if visible_text and mapped_meaning:
            return f'The sign says "{visible_text}." {mapped_meaning}'
        if mapped_meaning:
            return mapped_meaning
        if symbol:
            return f"This looks like a sign with {symbol}, but I am not certain what it means. Please hold it straighter or move a little closer."
        if intent == Intent.READ_SIGN_TEXT:
            task = "Read the sign text clearly. Do not add extra explanation unless needed."
        else:
            task = "Explain the sign or symbol meaning in plain language."
        prompt = f"""
{RULES}
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
        prompt = f"""
{RULES}
Detected kind: {kind}
Dialogue action: {intent.value}
Gemma image analysis JSON: {image_analysis}
Plant ID result: {plant_id}
If plant ID has a confident common name, say what it appears to be and give one short useful note.
If species is not confirmed, say the exact type is uncertain and offer to describe what is visible.
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
        return "I can see what may be a sign, but I cannot read or interpret it clearly. Please hold it straighter or move a little closer."

    return describe_plant(crop)


def _demo():
    assert "pedestrian crossing" in known_sign_meaning("yellow diamond", "walking person symbol")
    assert "pedestrian crossing" in known_sign_meaning("yellow diamond shape with a black silhouette of a person walking")
    assert "stop" in known_sign_meaning("STOP", "red octagon").lower()


if __name__ == "__main__":
    _demo()
