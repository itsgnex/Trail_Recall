from enum import Enum
import re

from .ollama_client import generate_json


class Intent(Enum):
    EXPLAIN_CURRENT_OBJECT = "EXPLAIN_CURRENT_OBJECT"
    READ_SIGN_TEXT = "READ_SIGN_TEXT"
    EXPLAIN_SIGN_MEANING = "EXPLAIN_SIGN_MEANING"
    IDENTIFY_PLANT = "IDENTIFY_PLANT"
    EXPLAIN_PLANT = "EXPLAIN_PLANT"
    MORE_DETAIL = "MORE_DETAIL"
    DESCRIBE_CURRENT_OBJECT = "DESCRIBE_CURRENT_OBJECT"
    IDENTIFY_CURRENT_OBJECT = "IDENTIFY_CURRENT_OBJECT"
    WHAT_AM_I_LOOKING_AT = "WHAT_AM_I_LOOKING_AT"
    CANCEL = "CANCEL"
    STOP_LISTENING = "STOP_LISTENING"
    REPEAT_LAST_MESSAGE = "REPEAT_LAST_MESSAGE"
    SPEAK_SLOWER = "SPEAK_SLOWER"
    GENERAL_QUESTION = "GENERAL_QUESTION"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"


def has(text, pattern):
    return re.search(pattern, text) is not None


def confirmation_action(detected_kind):
    if detected_kind == "plant":
        return Intent.EXPLAIN_PLANT
    if detected_kind == "sign":
        return Intent.EXPLAIN_SIGN_MEANING
    return Intent.WHAT_AM_I_LOOKING_AT


def is_help_prompt(text):
    text = (text or "").lower()
    return "would you like" in text or "should i" in text or "do you want me" in text or "would it help" in text


def is_confirmation_reply(text):
    text = f" {text.lower()} "
    return has(text, r"\b(you can go ahead|just go ahead|go ahead|go on|continue|please continue|sure go ahead|yeah go ahead|why not|yes|yeah|yep|sure|okay|ok|please do|can you do it|do it)\b")


def is_acknowledgement(text):
    text = f" {text.lower()} "
    return has(text, r"\b(thanks|thank you|thank u|thankyou|got it|gotcha|sounds good|all right|alright|okay thanks|ok thanks|thanks okay|thank you okay)\b")


def is_retry_request(text):
    text = f" {text.lower()} "
    return has(text, r"\b(try again|check again|look again|scan again|one more time)\b")


def is_image_request(text):
    text = f" {text.lower()} "
    return has(
        text,
        r"\b(what am i looking at|describe this|describe it|describe what i am looking at|what is this|what is that|what is in front of me|what do you see|in front of me|can you tell me what this is|can you tell me what that is|this sign|read this sign|what does this sign mean|what plant is this|tell me about this plant|read this)\b",
    )


def is_general_question(text):
    text = f" {text.lower()} "
    if is_image_request(text):
        return False
    return has(text, r"\b(what is|what are|who is|who are|why is|why are|how do|how does|tell me about)\b")


def classify_intent_fallback(text, detected_kind=None, last_question_type="none"):
    text = f" {text.lower()} "
    if is_retry_request(text):
        return Intent.IDENTIFY_CURRENT_OBJECT
    if has(text, r"\b(stop|that'?s enough|no more|be quiet|cancel)\b"):
        return Intent.STOP_LISTENING
    if has(text, r"\b(no|nope|not now|leave it|i'?m good|i am good|don't|do not)\b"):
        return Intent.CANCEL
    if has(text, r"\b(repeat|again|say that|one more time)\b"):
        return Intent.REPEAT_LAST_MESSAGE
    if has(text, r"\b(tell me more|more|give me more information|yes tell me more|what else|explain more)\b"):
        return Intent.MORE_DETAIL
    if has(text, r"\b(describe this|describe it|what am i looking at|what is this|what is in front of me|what do you see)\b"):
        return Intent.WHAT_AM_I_LOOKING_AT
    if has(text, r"\b(slower|slowly|slow down)\b"):
        return Intent.SPEAK_SLOWER
    if has(text, r"\b(read|what does it say|text|words)\b"):
        return Intent.READ_SIGN_TEXT if detected_kind == "sign" else Intent.EXPLAIN_CURRENT_OBJECT
    if has(text, r"\b(mean|means|meaning|explain the sign|what is this sign)\b"):
        return Intent.EXPLAIN_SIGN_MEANING if detected_kind == "sign" else Intent.EXPLAIN_CURRENT_OBJECT
    if has(text, r"\b(what plant is this|identify this plant|tell me about this plant)\b"):
        return Intent.IDENTIFY_PLANT
    if detected_kind == "plant" and has(text, r"\b(what is it|what is that|plant|flower|leaf|leaves)\b"):
        return Intent.EXPLAIN_PLANT
    if is_general_question(text):
        return Intent.GENERAL_QUESTION
    if is_acknowledgement(text):
        return Intent.CANCEL
    if is_confirmation_reply(text):
        if last_question_type in {"initial_permission", "follow_up_offer"}:
            return confirmation_action(detected_kind)
        if last_question_type == "more_detail_offer":
            return Intent.MORE_DETAIL
        if last_question_type == "retry_clearer_view":
            return Intent.IDENTIFY_CURRENT_OBJECT
        return Intent.ASK_CLARIFICATION
    return Intent.ASK_CLARIFICATION


def classify_intent(text, config=None):
    intent, _ = classify_intent_with_source(text, config)
    return intent


def classify_intent_with_source(text, config=None, detected_kind=None, last_message="", ocr_text="", clip_confidence=None, last_question_type="none"):
    if text and is_retry_request(text):
        return Intent.IDENTIFY_CURRENT_OBJECT, "rule"
    if text and has(text, r"\b(what plant is this|identify this plant|tell me about this plant)\b"):
        return Intent.IDENTIFY_PLANT, "rule"
    if text and has(text, r"\b(can you tell me what this is|can you tell me what that is|what is this|what is that|what is in front of me|describe what i am looking at|what do you see)\b"):
        return Intent.WHAT_AM_I_LOOKING_AT, "rule"
    if text and is_acknowledgement(text):
        return Intent.CANCEL, "rule"
    if text and is_general_question(text):
        return Intent.GENERAL_QUESTION, "rule"
    if text and last_question_type in {"initial_permission", "follow_up_offer"} and detected_kind in {"plant", "sign"} and is_help_prompt(last_message) and is_confirmation_reply(text):
        return confirmation_action(detected_kind), "rule"
    if text and last_question_type == "more_detail_offer" and is_confirmation_reply(text):
        return Intent.MORE_DETAIL, "rule"
    if text and last_question_type == "retry_clearer_view" and is_confirmation_reply(text):
        return Intent.IDENTIFY_CURRENT_OBJECT, "rule"
    if text and last_question_type in {"initial_permission", "follow_up_offer"} and is_confirmation_reply(text):
        return confirmation_action(detected_kind), "rule"
    if text and getattr(config, "use_llm_intent", False):
        model = getattr(config, "dialogue_model", "gemma3:1b")
        print(f"dialogue model: {model}")
        prompt = f"""
Return JSON only. No markdown. No explanation.
Allowed actions: EXPLAIN_CURRENT_OBJECT, READ_SIGN_TEXT, EXPLAIN_SIGN_MEANING, IDENTIFY_PLANT, EXPLAIN_PLANT, MORE_DETAIL, DESCRIBE_CURRENT_OBJECT, IDENTIFY_CURRENT_OBJECT, WHAT_AM_I_LOOKING_AT, CANCEL, STOP_LISTENING, REPEAT_LAST_MESSAGE, SPEAK_SLOWER, GENERAL_QUESTION, ASK_CLARIFICATION.
detected_kind={detected_kind or "unknown"}
assistant_prompt={last_message!r}
user_reply={text!r}
last_assistant_question_type={last_question_type!r}

Rules:
- yes/sure/why not/sure why not/just go ahead/go ahead/go on/continue/please continue/okay/yes please/can you do it/do it -> confirmation
- confirmation + detected_kind=plant -> EXPLAIN_PLANT
- confirmation + detected_kind=sign -> EXPLAIN_SIGN_MEANING
- sign + asks what it says -> READ_SIGN_TEXT
- sign + asks what it means -> EXPLAIN_SIGN_MEANING
- what plant is this / identify this plant / tell me about this plant -> IDENTIFY_PLANT
- plant + asks what it is -> EXPLAIN_PLANT
- general question -> GENERAL_QUESTION
- try again/check again/look again/scan again/one more time -> IDENTIFY_CURRENT_OBJECT
- tell me more/more/what else/explain more -> MORE_DETAIL
- what am I looking at/what is this/describe this/what do you see -> WHAT_AM_I_LOOKING_AT
- stop/that's enough/no more/be quiet/cancel -> STOP_LISTENING
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
user_reply="tell me more", detected_kind=plant -> {{"action":"MORE_DETAIL","target":"plant","should_continue":true,"confidence":0.9}}
user_reply="what am I looking at", detected_kind=other -> {{"action":"WHAT_AM_I_LOOKING_AT","target":"object","should_continue":true,"confidence":0.9}}
user_reply="what is photosynthesis", detected_kind=other -> {{"action":"GENERAL_QUESTION","target":"general","should_continue":false,"confidence":0.9}}

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
    return classify_intent_fallback(text or "", detected_kind, last_question_type), "fallback"


def _demo():
    assert classify_intent_fallback("Yes, what is it?", "plant", "initial_permission") == Intent.EXPLAIN_PLANT
    assert classify_intent_fallback("Can you read it?", "sign", "initial_permission") == Intent.READ_SIGN_TEXT
    assert classify_intent_fallback("What does that sign mean?", "sign") == Intent.EXPLAIN_SIGN_MEANING
    assert classify_intent_fallback("why not", "plant", "initial_permission") == Intent.EXPLAIN_PLANT
    assert classify_intent_fallback("why not", "sign", "initial_permission") == Intent.EXPLAIN_SIGN_MEANING
    assert classify_intent_fallback("just go ahead", "sign", "initial_permission") == Intent.EXPLAIN_SIGN_MEANING
    assert classify_intent_fallback("go on", "plant", "initial_permission") == Intent.EXPLAIN_PLANT
    assert classify_intent_fallback("you can go ahead", "plant", "more_detail_offer") == Intent.MORE_DETAIL
    assert classify_intent_fallback("can you do it", "sign", "initial_permission") == Intent.EXPLAIN_SIGN_MEANING
    assert classify_intent_fallback("why not", "plant", "follow_up_offer") == Intent.EXPLAIN_PLANT
    assert classify_intent_fallback("why not", "sign", "follow_up_offer") == Intent.EXPLAIN_SIGN_MEANING
    assert classify_intent_fallback("no not now", "plant") == Intent.CANCEL
    assert classify_intent_fallback("Repeat that", "sign") == Intent.REPEAT_LAST_MESSAGE
    assert classify_intent_fallback("tell me more", "plant") == Intent.MORE_DETAIL
    assert classify_intent_fallback("what am I looking at", "other") == Intent.WHAT_AM_I_LOOKING_AT
    assert classify_intent_fallback("what plant is this", "other") == Intent.IDENTIFY_PLANT
    assert classify_intent_fallback("what is photosynthesis", "other") == Intent.GENERAL_QUESTION
    assert classify_intent_fallback("stop", "sign") == Intent.STOP_LISTENING
    assert classify_intent_fallback("try again", "plant") == Intent.IDENTIFY_CURRENT_OBJECT
    assert confirmation_action("plant") == Intent.EXPLAIN_PLANT


if __name__ == "__main__":
    _demo()
