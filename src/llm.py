import json
import re
import time

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


def answer_general_question_with_1b(user_question, config=None):
    text = (user_question or "").strip()
    lower = text.lower()
    if not text:
        return ""
    if re.search(r"\b(weather|today|tonight|tomorrow|current|live|latest)\b", lower) or "right now" in lower:
        return "I cannot check live information right now, but I can answer general questions."
    if not getattr(config, "use_ollama", False):
        return "I can answer general questions, but I cannot check live information right now."

    model = getattr(config, "general_question_model", "gemma3:1b")
    prompt = f"""
{RULES}
Answer this general question in 1 to 3 short sentences.
No markdown. No internal model details. No live or current claims.
Question: {text}
If the question needs live or current information, say: I cannot check live information right now, but I can answer general questions.
"""
    print(f"general question model: {model}")
    return clean_spoken_answer(generate(prompt, config, timeout=getattr(config, "ollama_dialogue_timeout", 30), model=model) or "I cannot check live information right now, but I can answer general questions.")


def plant_id_payload(plant_id):
    if isinstance(plant_id, PlantIdResult):
        return {
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
    return plant_id


def clean_spoken_answer(text):
    if not text:
        return text
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.split(r"\b(would you like|would it help|do you want me to|can i help with)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    cleaned = re.sub(r"\s*\?\s*$", "", cleaned).strip()
    cleaned = cleaned.rstrip(" ?.")
    return cleaned + "." if cleaned and not cleaned.endswith(".") else cleaned


def _polite_sign_sentence(text):
    cleaned = clean_spoken_answer(text)
    if not cleaned:
        return cleaned
    lower = cleaned.lower()
    if lower.startswith(("this ", "the ", "it ", "you ", "i ", "please ")):
        return cleaned
    return clean_spoken_answer(f"This appears to be a sign. It means {cleaned}")


def known_sign_meaning(*parts):
    text = " ".join(str(part or "").lower() for part in parts)
    mappings = [
        (("walking person", "person walking", "person crossing", "pedestrian"), "This appears to be a pedestrian crossing sign. Please watch carefully for people walking or crossing nearby."),
        (("stop", "red octagon"), "This appears to be a stop sign. Please stop here and check carefully for traffic or pedestrians before continuing."),
        (("one way",), "This appears to be a one-way sign. Please follow the direction shown."),
        (("left arrow", "arrow left"), "This sign points left. Please follow or pay attention to the direction shown."),
        (("right arrow", "arrow right"), "This sign points right. Please follow or pay attention to the direction shown."),
        (("merge", "lanes merge", "two lanes merge"), "This appears to be a lane merge warning sign. Please be ready to merge carefully."),
        (("no entry", "do not enter"), "This appears to be a do-not-enter sign. Please do not go that way."),
        (("exit",), "This appears to be an exit sign. It points toward a way out."),
        (("caution", "warning triangle", "warning"), "This appears to be a warning sign. Please use caution nearby."),
        (("nutrition facts", "serving size", "calories", "protein", "carbohydrate", "sugar", "sodium"), "This appears to be a nutrition facts label. It lists serving size, calories, sugar, and other nutrition information."),
        (("barcode",), "This appears to be a barcode label. It is used to identify the item."),
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
    recommended_action = (vision_result.get("recommended_action") or "").strip()
    is_clear = bool(vision_result.get("is_clear_enough", True))
    confidence = float(vision_result.get("confidence") or 0)

    if not is_clear or confidence < 0.35:
        return None

    text = " ".join(part.lower() for part in (visible_text, symbol, description, plain_meaning, recommended_action) if part)
    if "stop" in text:
        return "This appears to be a stop sign. Please stop here and check carefully for traffic or pedestrians before continuing."
    if "pedestrian" in text:
        return "This appears to be a pedestrian crossing sign. Please watch carefully for people walking or crossing nearby."
    if "one way" in text:
        return "This appears to be a one-way sign. Please follow the direction shown."
    if "do not enter" in text or "no entry" in text:
        return "This appears to be a do-not-enter sign. Please do not go that way."
    if "merge" in text:
        return "This appears to be a lane merge warning sign. Please be ready to merge carefully."
    if "warning" in text or "caution" in text:
        return "This appears to be a warning sign. Please use caution nearby."
    if "exit" in text:
        return "This appears to be an exit sign. It points toward a way out."

    if plain_meaning:
        return _polite_sign_sentence(plain_meaning)

    meaning = known_sign_meaning(visible_text, symbol, description)
    if not meaning:
        return None

    if visible_text and "stop" in visible_text.lower():
        lead = "This appears to be a stop sign. Please stop here and check carefully before continuing."
    elif visible_text and "one way" in visible_text.lower():
        lead = "This appears to be a one-way sign. Please follow the direction shown."
    elif "pedestrian" in meaning.lower():
        lead = "This appears to be a pedestrian crossing sign."
    elif "lane merge" in meaning.lower():
        lead = "This appears to be a lane merge warning sign."
    elif "do-not-enter" in meaning.lower():
        lead = "This appears to be a do-not-enter sign."
    elif "exit" in meaning.lower():
        lead = "This appears to be an exit sign."
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
        return clean_spoken_answer(lead)
    return clean_spoken_answer(f"{lead} It means {cleaned}")


def _response_cache_key(kind, intent, image_analysis=None, ocr_text="", plant_id=None):
    payload = {
        "kind": kind,
        "intent": getattr(intent, "value", intent),
        "ocr_text": ocr_text or "",
    }
    if image_analysis:
        payload["image_type"] = image_analysis.get("image_type")
        payload["visible_text"] = image_analysis.get("visible_text")
        payload["symbol_or_icon"] = image_analysis.get("symbol_or_icon")
        payload["description"] = image_analysis.get("description")
        payload["plain_meaning"] = image_analysis.get("plain_meaning")
        payload["recommended_action"] = image_analysis.get("recommended_action")
        payload["confidence"] = image_analysis.get("confidence")
    if plant_id is not None:
        payload["plant_common_name"] = getattr(plant_id, "common_name", None)
        payload["plant_scientific_name"] = getattr(plant_id, "scientific_name", None)
        payload["plant_score"] = getattr(plant_id, "score", None)
    return json.dumps(payload, sort_keys=True, default=str)


def _get_cached_response(state, cache_key, ttl_seconds=120):
    if state is None:
        return None
    if getattr(state, "last_response_cache_key", None) != cache_key:
        return None
    age = time.monotonic() - float(getattr(state, "last_response_cache_time", 0.0) or 0.0)
    if age > ttl_seconds:
        return None
    return getattr(state, "last_response_cache_value", None)


def _store_cached_response(state, cache_key, answer):
    if state is None:
        return
    state.last_response_cache_key = cache_key
    state.last_response_cache_value = answer
    state.last_response_cache_time = time.monotonic()


def answer_for(kind, crop, intent, config=None, state=None):
    fallback = fallback_answer(kind, crop, intent)
    if not getattr(config, "use_ollama", False):
        return fallback

    ocr_text = read_text(crop) if kind == "sign" else ""
    image_analysis = analyze_image_with_gemma(crop, kind, intent.value, config, ocr_text)
    if state is not None:
        state.last_vision_result = image_analysis
        state.last_plant_id_result = None

    cache_key = _response_cache_key(kind, intent, image_analysis, ocr_text)
    cached = _get_cached_response(state, cache_key, ttl_seconds=120)
    if cached:
        return cached

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
            return _polite_sign_sentence(plain_meaning)
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
Do not start with labels like Warning, Description, Meaning, or Recommended action. Speak as a natural assistant. Use one or two short sentences. Avoid repeating the same idea. Do not ask a follow-up question. Do not end with Would you like...
Do not end with a question or invite the user to ask for more.
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
        vision_label = (image_analysis or {}).get("image_type") or "uncertain"
        vision_confidence = float((image_analysis or {}).get("confidence") or 0.0)
        vision_description = (image_analysis or {}).get("description") or ""
        print(f"vision label: {vision_label}")
        print(f"vision confidence: {vision_confidence:.2f}")
        if vision_label != "plant" or vision_confidence < 0.55:
            print("plantnet: skipped")
            answer = "I don't see a clear plant in the focus area. Please point the camera directly at leaves or flowers."
            if state is not None:
                state.last_plant_id_result = None
            _store_cached_response(state, cache_key, answer)
            return answer

        print("plantnet: plant confirmed by vision model")
        if not getattr(config, "use_plant_id", False):
            answer = clean_spoken_answer(f"This appears to be a plant. {vision_description}" if vision_description else "This appears to be a plant, but I cannot identify the exact type yet.")
            if state is not None:
                state.last_plant_id_result = None
            _store_cached_response(state, cache_key, answer)
            return answer

        print("plantnet: called")
        plant_id = identify_plant(crop, config)
        if state is not None:
            state.last_plant_id_result = plant_id
        if isinstance(plant_id, PlantIdResult):
            if plant_id.success and plant_id.confidence_level == "high" and plant_id.common_name:
                answer = f"This appears to be {plant_id.common_name}. It has features that match that plant fairly well."
            elif plant_id.success and plant_id.confidence_level == "medium" and plant_id.common_name:
                answer = f"This may be {plant_id.common_name}, but I am not fully certain. The image looks similar to that plant."
            else:
                answer = "This appears to be a plant, but I am not certain of the exact type from this image. A closer view of the leaves or flowers would help."
        else:
            answer = fallback
        _store_cached_response(state, cache_key, answer)
        return clean_spoken_answer(answer)


def generate_more_detail_response(state, config):
    kind = state.last_detected_kind or "object"
    plant_id = state.last_plant_id_result
    if kind == "plant" and isinstance(plant_id, PlantIdResult):
        if plant_id.success and plant_id.confidence_level == "high" and plant_id.common_name:
            prompt = f"""
{RULES}
Give one extra simple fact about this plant in one short sentence.
Plant name: {plant_id.common_name}
Scientific name: {plant_id.scientific_name}
Family: {plant_id.family}
Keep it short, safe, and concrete. Do not speculate about care, toxicity, or medicinal use.
Do not ask a follow-up question.
            """
            model = getattr(config, "final_response_model", "gemma3:4b")
            return clean_spoken_answer(generate(prompt, config, timeout=getattr(config, "ollama_text_timeout", 60), model=model) or f"{plant_id.common_name} matches the image fairly well.")
        return "I do not have a reliable plant name yet. A closer image of the leaves or flowers would help."

    prompt = f"""
{RULES}
Give one short follow-up detail about the same {kind}.
Last action: {state.last_action}
Last answer: {state.last_final_answer}
Vision result: {state.last_vision_result}
Plant ID result: {plant_id_payload(state.last_plant_id_result)}
For plants, give 1-2 useful facts without medical, toxicity, or care claims unless clearly supported.
For signs, explain the meaning slightly more clearly.
For unclear objects, describe only what is visible.
Do not ask a follow-up question.
"""
    model = getattr(config, "final_response_model", "gemma3:4b")
    return clean_spoken_answer(generate(prompt, config, timeout=getattr(config, "ollama_text_timeout", 60), model=model) or "I do not have enough clear detail to add more right now.")


def describe_current_object(crop, config, state=None):
    image_analysis = analyze_image_with_gemma(crop, "object", "WHAT_AM_I_LOOKING_AT", config)
    plant_id = None
    image_type = (image_analysis or {}).get("image_type")
    if image_type == "plant" and getattr(config, "use_plant_id", False):
        plant_id = identify_plant(crop, config)
    if state is not None:
        state.last_detected_kind = image_type or "object"
        state.last_vision_result = image_analysis
        state.last_plant_id_result = plant_id

    cache_key = _response_cache_key("object", Intent.WHAT_AM_I_LOOKING_AT, image_analysis, plant_id=plant_id)
    cached = _get_cached_response(state, cache_key, ttl_seconds=120)
    if cached:
        return cached

    if image_type == "plant":
        if getattr(config, "use_plant_id", False) and isinstance(plant_id, PlantIdResult):
            if plant_id.success and plant_id.confidence_level == "high" and plant_id.common_name:
                answer = f"This appears to be {plant_id.common_name}. It has features that match that plant fairly well."
            elif plant_id.success and plant_id.confidence_level == "medium" and plant_id.common_name:
                answer = f"This may be {plant_id.common_name}, but I am not fully certain. The image looks similar to that plant."
            else:
                answer = "This appears to be a plant, but I am not certain of the exact type from this image. A closer view of the leaves or flowers would help."
        else:
            vision_description = (image_analysis or {}).get("description") or ""
            answer = clean_spoken_answer(f"This appears to be a plant. {vision_description}" if vision_description else "This appears to be a plant, but I cannot identify the exact type yet.")
        _store_cached_response(state, cache_key, answer)
        return answer

    if image_type in {"sign", "symbol_sign", "text_sign"}:
        natural = naturalize_sign_response(image_analysis)
        if natural:
            _store_cached_response(state, cache_key, natural)
            return natural
    prompt = f"""
{RULES}
Describe the main object in the center focus crop in one or two short sentences.
Only use the image analysis and Plant ID result below.
Image analysis: {image_analysis}
Plant ID result: {plant_id_payload(plant_id)}
If unclear, ask the user to hold it steadier or move closer.
Do not ask a follow-up question.
"""
    model = getattr(config, "final_response_model", "gemma3:4b")
    answer = clean_spoken_answer(generate(prompt, config, timeout=getattr(config, "ollama_text_timeout", 60), model=model) or "I can see something in the center, but it is not clear enough yet. Try holding it steadier or moving a little closer.")
    _store_cached_response(state, cache_key, answer)
    return answer


def fallback_answer(kind, crop, intent):
    if kind == "sign":
        text = read_text(crop)
        if text:
            if intent == Intent.READ_SIGN_TEXT:
                return f"The sign says: {text}."
            return f"This appears to be a sign with the text: {text}."
        return get_unclear_sign_response()

    return describe_plant(crop) or get_unclear_plant_response()


def _demo():
    assert "pedestrian crossing" in known_sign_meaning("yellow diamond", "walking person symbol")
    assert "pedestrian crossing" in known_sign_meaning("yellow diamond shape with a black silhouette of a person walking")
    assert "stop" in known_sign_meaning("STOP", "red octagon").lower()


if __name__ == "__main__":
    _demo()
