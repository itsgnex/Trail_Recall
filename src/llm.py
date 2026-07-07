import re
import time

from .intent import Intent
from .ollama_client import generate
from .openrouter_client import generate as generate_openrouter
from .ocr import read_text
from .plant_id import PlantIdResult, describe_plant, identify_plant
from .phrases import (
    get_clarification_response,
    get_repeat_response,
    get_unclear_plant_response,
    get_unclear_sign_response,
)
from .scene_memory import lookup_scene_memory, store_scene_memory
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
    return clean_spoken_answer(generate(prompt, config, timeout=getattr(config, "ollama_dialogue_timeout", 12), model=model) or "I cannot check live information right now, but I can answer general questions.")


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


def _plant_summary(plant_id):
    if not isinstance(plant_id, PlantIdResult):
        return ""
    parts = []
    if plant_id.common_name:
        parts.append(plant_id.common_name)
    if plant_id.scientific_name:
        parts.append(f"scientific name {plant_id.scientific_name}")
    if plant_id.family:
        parts.append(f"family {plant_id.family}")
    if plant_id.genus:
        parts.append(f"genus {plant_id.genus}")
    return ", ".join(parts)


def _plant_response_template(plant_id):
    if not isinstance(plant_id, PlantIdResult) or not plant_id.success:
        return "This appears to be a plant, but I am not certain of the exact type from this view."

    if plant_id.confidence_level == "high" and plant_id.common_name:
        return clean_spoken_answer(f"This looks like {plant_id.common_name}. That seems like a good match.")
    elif plant_id.confidence_level == "medium" and plant_id.common_name:
        return clean_spoken_answer(f"This may be {plant_id.common_name}. It is a reasonable match, but I am not completely certain.")
    elif plant_id.common_name:
        return clean_spoken_answer(f"This might be {plant_id.common_name}, but I am not very confident yet.")
    elif plant_id.scientific_name:
        return clean_spoken_answer(f"This may be {plant_id.scientific_name}, but I am not fully certain.")
    return "This appears to be a plant, but I am not certain of the exact type from this view."


def plant_response_from_id(plant_id, config=None):
    fallback = _plant_response_template(plant_id)
    if not isinstance(plant_id, PlantIdResult) or not plant_id.success:
        return fallback
    if not getattr(config, "plant_response_ai", True) or not getattr(config, "use_ollama", False):
        return fallback

    model = getattr(config, "final_response_model", "gemma3:1b")
    prompt = f"""
{RULES}
Write a natural spoken plant answer.
Speak as a helpful assistant for an older adult with memory difficulty.
Do not mention APIs, databases, confidence scores, JSON, PlantNet, or image analysis.
Do not sound like a form or a database readout.
Do not list taxonomy mechanically. Use the scientific name, family, or genus only if it sounds natural and useful.
Use 2 to 4 short sentences.
Use a warm, simple, human style.
You may ask one short follow-up question at the end if it would help the person.
If you ask a follow-up, only ask whether they want one more simple detail about the identification.
Do not ask about care, placement, watering, light, food safety, medicine, toxicity, or shopping.
Use only the facts below. Do not invent appearance, leaf colors, care, origin, toxicity, edibility, or why it has a name.
Do not use markdown, bullet points, labels, or italics.
Do not say actually.

Facts:
- common name: {plant_id.common_name or ""}
- scientific name: {plant_id.scientific_name or ""}
- family: {plant_id.family or ""}
- genus: {plant_id.genus or ""}
- match strength: {plant_id.confidence_level}
"""
    answer = ""
    if getattr(config, "openrouter_api_key", ""):
        openrouter_model = getattr(config, "openrouter_model", "google/gemini-3.1-flash-lite")
        print(f"plant response model: OpenRouter {openrouter_model}")
        answer = generate_openrouter(
            prompt,
            config,
            timeout=min(getattr(config, "openrouter_timeout", 8), 4),
            model=openrouter_model,
        )
    if not answer:
        print(f"plant response model: {model}")
        answer = generate(prompt, config, timeout=getattr(config, "ollama_dialogue_timeout", 12), model=model)
    return clean_spoken_answer(answer or fallback, strip_followups=False)


def clean_spoken_answer(text, strip_followups=True):
    if not text:
        return text
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.replace("*", "")
    cleaned = re.sub(r"\b(the|this|that)\s+(image|picture|photo)\b", r"\1 view", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bimage\b|\bpicture\b|\bphoto\b", "view", cleaned, flags=re.IGNORECASE)
    if strip_followups:
        cleaned = re.split(r"\b(if you would like|would you like|would it help|do you want me to|can i help with)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        cleaned = re.sub(r"\s*\?\s*$", "", cleaned).strip()
    if strip_followups:
        cleaned = cleaned.rstrip(" ?.")
    else:
        cleaned = cleaned.strip()
    if cleaned and not cleaned.endswith((".", "?")):
        return cleaned + "."
    return cleaned


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


def short_circuit_sign_answer(ocr_text, intent):
    text = (ocr_text or "").strip()
    if not text:
        return None
    meaning = known_sign_meaning(text)
    if intent == Intent.READ_SIGN_TEXT:
        if len(text) >= 4:
            return clean_spoken_answer(f"The sign says: {text}")
        if meaning:
            return meaning
        return None
    if meaning:
        return meaning
    if len(text) >= 8:
        return clean_spoken_answer(f"The sign says: {text}")
    return None


def _vision_sign_answer(crop, intent, config, ocr_text=""):
    image_analysis = analyze_image_with_gemma(crop, "sign", intent.value, config, ocr_text)
    natural = naturalize_sign_response(image_analysis)
    if natural:
        return natural, image_analysis
    visible_text = (image_analysis or {}).get("visible_text") or ""
    if intent == Intent.READ_SIGN_TEXT and visible_text:
        return clean_spoken_answer(f"The sign says: {visible_text}"), image_analysis
    return get_unclear_sign_response(), image_analysis


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


def _restore_plant_id(payload):
    if not isinstance(payload, dict):
        return None
    try:
        return PlantIdResult(
            provider=payload.get("provider", "plantnet"),
            success=bool(payload.get("success", False)),
            common_name=payload.get("common_name"),
            scientific_name=payload.get("scientific_name"),
            family=payload.get("family"),
            genus=payload.get("genus"),
            score=float(payload.get("score", 0.0) or 0.0),
            raw_top_result=payload.get("raw_top_result"),
            error=payload.get("error"),
        )
    except Exception:
        return None


def _restore_cached_scene(state, record):
    if not record:
        return None
    if state is not None:
        state.last_vision_result = record.get("vision_result")
        state.last_plant_id_result = _restore_plant_id(record.get("plant_id_result"))
    return record.get("answer")


def _get_cached_response(state, kind, intent, crop, ttl_seconds=120):
    record = lookup_scene_memory(kind, intent, crop, ttl_seconds=ttl_seconds)
    if record is None:
        return None
    return _restore_cached_scene(state, record)


def _store_cached_response(state, kind, intent, crop, answer):
    if state is None or not answer:
        return
    shared_kwargs = dict(
        crop_path=getattr(state, "last_crop_path", None),
        vision_result=getattr(state, "last_vision_result", None),
        plant_id_result=plant_id_payload(getattr(state, "last_plant_id_result", None)),
        ttl_seconds=3600,
        max_items=120,
    )
    store_scene_memory(
        kind,
        intent,
        crop,
        answer,
        **shared_kwargs,
    )
    store_scene_memory(
        kind,
        "scene",
        crop,
        answer,
        **shared_kwargs,
    )


def answer_for(kind, crop, intent, config=None, state=None):
    if not getattr(config, "use_ollama", False):
        return fallback_answer(kind, crop, intent)

    cached = _get_cached_response(state, kind, intent, crop, ttl_seconds=120)
    if cached:
        return cached

    ocr_text = read_text(crop) if kind == "sign" else ""
    if kind == "sign":
        easy_answer = short_circuit_sign_answer(ocr_text, intent)
        if easy_answer:
            if state is not None:
                state.last_vision_result = None
                state.last_plant_id_result = None
            _store_cached_response(state, kind, intent, crop, easy_answer)
            return easy_answer
        if intent == Intent.EXPLAIN_SIGN_MEANING or intent == Intent.EXPLAIN_CURRENT_OBJECT:
            answer, image_analysis = _vision_sign_answer(crop, intent, config, ocr_text)
            if state is not None:
                state.last_vision_result = image_analysis
                state.last_plant_id_result = None
            _store_cached_response(state, kind, intent, crop, answer)
            return answer
        answer = get_unclear_sign_response()
        _store_cached_response(state, kind, intent, crop, answer)
        return answer

    if kind == "plant":
        if not getattr(config, "use_plant_id", False):
            image_analysis = analyze_image_with_gemma(crop, kind, intent.value, config, ocr_text)
            if state is not None:
                state.last_vision_result = image_analysis
                state.last_plant_id_result = None
            vision_description = (image_analysis or {}).get("description") or ""
            answer = clean_spoken_answer(f"This appears to be a plant. {vision_description}" if vision_description else "This appears to be a plant, but I cannot identify the exact type yet.")
            _store_cached_response(state, kind, intent, crop, answer)
            return answer

        print("plantnet: called")
        plant_id = identify_plant(crop, config)
        if state is not None:
            state.last_plant_id_result = plant_id
            state.last_vision_result = None
        answer = plant_response_from_id(plant_id, config)
        _store_cached_response(state, kind, intent, crop, answer)
        return answer

    image_analysis = analyze_image_with_gemma(crop, kind, intent.value, config, ocr_text)
    if state is not None:
        state.last_vision_result = image_analysis
        state.last_plant_id_result = None

    image_type = (image_analysis or {}).get("image_type")
    if image_type == "plant":
        plant_id = None
        if getattr(config, "use_plant_id", False):
            plant_id = identify_plant(crop, config)
            if state is not None:
                state.last_plant_id_result = plant_id
            answer = plant_response_from_id(plant_id, config)
        else:
            vision_description = (image_analysis or {}).get("description") or ""
            answer = clean_spoken_answer(f"This appears to be a plant. {vision_description}" if vision_description else "This appears to be a plant, but I cannot identify the exact type yet.")
        _store_cached_response(state, kind, intent, crop, answer)
        return answer

    if image_type in {"sign", "symbol_sign", "text_sign"}:
        natural = naturalize_sign_response(image_analysis)
        if natural:
            _store_cached_response(state, kind, intent, crop, natural)
            return natural

    plant_id = None
    prompt = f"""
{RULES}
Describe the main object in the center focus crop in one or two short sentences.
Only use the image analysis and Plant ID result below.
Image analysis: {image_analysis}
Plant ID result: {plant_id_payload(plant_id)}
If unclear, ask the user to hold it steadier or move closer.
Do not ask a follow-up question.
    """
    model = getattr(config, "final_response_model", "gemma3:1b")
    answer = clean_spoken_answer(generate(prompt, config, timeout=getattr(config, "ollama_text_timeout", 20), model=model) or "I can see something in the center, but it is not clear enough yet. Try holding it steadier or moving a little closer.")
    _store_cached_response(state, kind, intent, crop, answer)
    return answer


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
            model = getattr(config, "final_response_model", "gemma3:1b")
            return clean_spoken_answer(generate(prompt, config, timeout=getattr(config, "ollama_text_timeout", 20), model=model) or f"{plant_id.common_name} matches the image fairly well.")
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
    model = getattr(config, "final_response_model", "gemma3:1b")
    return clean_spoken_answer(generate(prompt, config, timeout=getattr(config, "ollama_text_timeout", 20), model=model) or "I do not have enough clear detail to add more right now.")


def describe_current_object(crop, config, state=None):
    cached = _get_cached_response(state, "object", Intent.WHAT_AM_I_LOOKING_AT, crop, ttl_seconds=120)
    if cached:
        return cached

    image_analysis = analyze_image_with_gemma(crop, "object", "WHAT_AM_I_LOOKING_AT", config)
    plant_id = None
    image_type = (image_analysis or {}).get("image_type")
    if image_type == "plant" and getattr(config, "use_plant_id", False):
        plant_id = identify_plant(crop, config)
    if state is not None:
        state.last_detected_kind = image_type or "object"
        state.last_vision_result = image_analysis
        state.last_plant_id_result = plant_id

    if image_type == "plant":
        if getattr(config, "use_plant_id", False):
            if not isinstance(plant_id, PlantIdResult):
                plant_id = identify_plant(crop, config)
                if state is not None:
                    state.last_plant_id_result = plant_id
            answer = plant_response_from_id(plant_id, config)
        else:
            vision_description = (image_analysis or {}).get("description") or ""
            answer = clean_spoken_answer(f"This appears to be a plant. {vision_description}" if vision_description else "This appears to be a plant, but I cannot identify the exact type yet.")
        _store_cached_response(state, "object", Intent.WHAT_AM_I_LOOKING_AT, crop, answer)
        return answer

    if image_type in {"sign", "symbol_sign", "text_sign"}:
        natural = naturalize_sign_response(image_analysis)
        if natural:
            _store_cached_response(state, "object", Intent.WHAT_AM_I_LOOKING_AT, crop, natural)
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
    model = getattr(config, "final_response_model", "gemma3:1b")
    answer = clean_spoken_answer(generate(prompt, config, timeout=getattr(config, "ollama_text_timeout", 20), model=model) or "I can see something in the center, but it is not clear enough yet. Try holding it steadier or moving a little closer.")
    _store_cached_response(state, "object", Intent.WHAT_AM_I_LOOKING_AT, crop, answer)
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
