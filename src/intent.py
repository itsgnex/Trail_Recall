from enum import Enum
import re

from .ollama_client import generate_json


class Intent(Enum):
    EXPLAIN_CURRENT_OBJECT = "EXPLAIN_CURRENT_OBJECT"
    READ_SIGN_TEXT = "READ_SIGN_TEXT"
    EXPLAIN_SIGN_MEANING = "EXPLAIN_SIGN_MEANING"
    EXPLAIN_PLANT = "EXPLAIN_PLANT"
    CANCEL = "CANCEL"
    REPEAT_LAST_MESSAGE = "REPEAT_LAST_MESSAGE"
    SPEAK_SLOWER = "SPEAK_SLOWER"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"


def has(text, pattern):
    return re.search(pattern, text) is not None


def confirmation_action(detected_kind):
    if detected_kind == "plant":
        return Intent.EXPLAIN_PLANT
    if detected_kind == "sign":
        return Intent.EXPLAIN_SIGN_MEANING
    return Intent.EXPLAIN_CURRENT_OBJECT


def is_help_prompt(text):
    text = (text or "").lower()
    return "would you like" in text or "should i" in text or "do you want me" in text or "would it help" in text


def is_confirmation_reply(text):
    text = f" {text.lower()} "
    return has(text, r"\b(just go ahead|go ahead|go on|continue|please continue|sure go ahead|yeah go ahead|why not|yes|yeah|yep|sure|okay|ok|please do|can you do it|do it)\b")


def classify_intent_fallback(text, detected_kind=None):
    text = f" {text.lower()} "
    if has(text, r"\b(no|nope|stop|cancel|not now|leave it|i'?m good|i am good|don't|do not)\b"):
        return Intent.CANCEL
    if has(text, r"\b(repeat|again|say that|one more time)\b"):
        return Intent.REPEAT_LAST_MESSAGE
    if has(text, r"\b(slower|slowly|slow down)\b"):
        return Intent.SPEAK_SLOWER
    if has(text, r"\b(read|what does it say|text|words)\b"):
        return Intent.READ_SIGN_TEXT if detected_kind == "sign" else Intent.EXPLAIN_CURRENT_OBJECT
    if has(text, r"\b(mean|means|meaning|explain the sign|what is this sign)\b"):
        return Intent.EXPLAIN_SIGN_MEANING if detected_kind == "sign" else Intent.EXPLAIN_CURRENT_OBJECT
    if detected_kind == "plant" and has(text, r"\b(what is it|what is that|plant|flower|leaf|leaves)\b"):
        return Intent.EXPLAIN_PLANT
    if is_confirmation_reply(text):
        return confirmation_action(detected_kind)
    return Intent.ASK_CLARIFICATION


def classify_intent(text, config=None):
    intent, _ = classify_intent_with_source(text, config)
    return intent


def classify_intent_with_source(text, config=None, detected_kind=None, last_message="", ocr_text="", clip_confidence=None):
    if text and detected_kind in {"plant", "sign"} and is_help_prompt(last_message) and is_confirmation_reply(text):
        return confirmation_action(detected_kind), "rule"
    if text and getattr(config, "use_llm_intent", False):
        model = getattr(config, "dialogue_model", "gemma3:1b")
        print(f"dialogue model: {model}")
        prompt = f"""
Return JSON only. No markdown. No explanation.
Allowed actions: EXPLAIN_CURRENT_OBJECT, READ_SIGN_TEXT, EXPLAIN_SIGN_MEANING, EXPLAIN_PLANT, CANCEL, REPEAT_LAST_MESSAGE, SPEAK_SLOWER, ASK_CLARIFICATION.
detected_kind={detected_kind or "unknown"}
assistant_prompt={last_message!r}
user_reply={text!r}

Rules:
- yes/sure/why not/sure why not/just go ahead/go ahead/go on/continue/please continue/okay/yes please/can you do it/do it -> confirmation
- confirmation + detected_kind=plant -> EXPLAIN_PLANT
- confirmation + detected_kind=sign -> EXPLAIN_SIGN_MEANING
- sign + asks what it says -> READ_SIGN_TEXT
- sign + asks what it means -> EXPLAIN_SIGN_MEANING
- plant + asks what it is -> EXPLAIN_PLANT
- refusal -> CANCEL
- repeat -> REPEAT_LAST_MESSAGE
- slower -> SPEAK_SLOWER
- unclear -> ASK_CLARIFICATION

Examples:
user_reply="why not", detected_kind=plant -> {{"action":"EXPLAIN_PLANT","target":"plant","should_continue":true,"confidence":0.9}}
user_reply="why not", detected_kind=sign -> {{"action":"EXPLAIN_SIGN_MEANING","target":"sign","should_continue":true,"confidence":0.9}}
user_reply="what does it mean", detected_kind=sign -> {{"action":"EXPLAIN_SIGN_MEANING","target":"sign","should_continue":true,"confidence":0.9}}
user_reply="what does it say", detected_kind=sign -> {{"action":"READ_SIGN_TEXT","target":"sign","should_continue":true,"confidence":0.9}}
user_reply="no not now", detected_kind=plant -> {{"action":"CANCEL","target":"plant","should_continue":false,"confidence":0.9}}

Schema:
{{"action":"EXPLAIN_CURRENT_OBJECT","target":"{detected_kind or 'object'}","should_continue":true,"confidence":0.9}}
"""
        data = generate_json(prompt, config, timeout=getattr(config, "ollama_dialogue_timeout", 30), model=model)
        try:
            intent = Intent(data.get("action") or data.get("intent"))
            confidence = float(data.get("confidence", 0))
            if confidence >= 0.45:
                return intent, model
        except Exception:
            pass
    return classify_intent_fallback(text or "", detected_kind), "fallback"


def _demo():
    assert classify_intent_fallback("Yes, what is it?", "plant") == Intent.EXPLAIN_PLANT
    assert classify_intent_fallback("Can you read it?", "sign") == Intent.READ_SIGN_TEXT
    assert classify_intent_fallback("What does that sign mean?", "sign") == Intent.EXPLAIN_SIGN_MEANING
    assert classify_intent_fallback("why not", "plant") in {Intent.EXPLAIN_PLANT, Intent.EXPLAIN_CURRENT_OBJECT}
    assert classify_intent_fallback("why not", "sign") in {Intent.EXPLAIN_SIGN_MEANING, Intent.READ_SIGN_TEXT}
    assert classify_intent_fallback("just go ahead", "sign") == Intent.EXPLAIN_SIGN_MEANING
    assert classify_intent_fallback("go on", "plant") == Intent.EXPLAIN_PLANT
    assert classify_intent_fallback("sure why not", "plant") in {Intent.EXPLAIN_PLANT, Intent.EXPLAIN_CURRENT_OBJECT}
    assert classify_intent_fallback("can you do it", "sign") in {Intent.EXPLAIN_SIGN_MEANING, Intent.READ_SIGN_TEXT}
    assert classify_intent_fallback("no not now", "plant") == Intent.CANCEL
    assert classify_intent_fallback("Repeat that", "sign") == Intent.REPEAT_LAST_MESSAGE
    assert confirmation_action("plant") == Intent.EXPLAIN_PLANT


if __name__ == "__main__":
    _demo()
