from enum import Enum
import json
import re

from .openrouter_client import generate as generate_openrouter


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
    START_TRAIL = "START_TRAIL"
    STOP_TRAIL = "STOP_TRAIL"
    NAVIGATE_BACK = "NAVIGATE_BACK"
    DESTINATION_REACHED = "DESTINATION_REACHED"
    CHOOSE_LEFT = "CHOOSE_LEFT"
    CHOOSE_RIGHT = "CHOOSE_RIGHT"
    CHOOSE_SAVED_ROUTE = "CHOOSE_SAVED_ROUTE"
    CHOOSE_ALTERNATE_ROUTE = "CHOOSE_ALTERNATE_ROUTE"
    GENERAL_QUESTION = "GENERAL_QUESTION"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"


GEMINI_INTENT_CONFIDENCE = 0.89
GEMINI_INTENTS = frozenset(intent.value for intent in Intent if intent != Intent.ASK_CLARIFICATION)


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
    return has(text, r"\b(that would be great|would be great|that'd be great|that would help|would help|please do|yes please|please|you can go ahead|just go ahead|go ahead|go on|continue|please continue|sure go ahead|yeah go ahead|why not|yes|yeah|yea|yep|sure|okay|ok|can you do it|can u do it|do it)\b")


def infer_permission_reply(text):
    text = f" {text.lower()} "
    if has(text, r"\b(stop|cancel|be quiet|no more|that'?s enough)\b"):
        return "stop"
    if has(text, r"\b(no|nope|nah|not now|later|maybe later|leave it|don't|do not|i'?m good|i am good)\b"):
        return "no"
    if has(text, r"\b(repeat|say that)\b"):
        return "repeat"
    if has(text, r"\b(sounds good|sounds useful|that sounds useful|that helps|that would help|that would be helpful|i'?d like that|i would like that|i'?d appreciate that|i would appreciate that|good idea|fine|alright|all right|go for it|let'?s do it|do that|tell me|show me|help with it|check it|explain it|be great|great|play store|play george)\b"):
        return "yes"
    if is_confirmation_reply(text):
        return "yes"
    return ""


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
        r"\b(help me|can you help me|what am i looking at|describe this|describe it|describe what i am looking at|what is this|what is that|what is in front of me|what do you see|in front of me|can you tell me what this is|can you tell me what that is|this sign|read this sign|what does this sign mean|what plant is this|tell me about this plant|read this)\b",
    )


def is_general_question(text):
    text = f" {text.lower()} "
    if is_image_request(text):
        return False
    return has(text, r"\b(what is|what are|who is|who are|why is|why are|how do|how does|tell me about)\b")


def is_trail_command(text):
    text = f" {_normalize_trail_text(text)} "
    if re.search(r"\b(start|begin|record)\s+((my|the)\s+)?(route|trail|trails|tracking)\b", text):
        return Intent.START_TRAIL
    if re.search(r"\b(stop|end|finish)\s+((my|the)\s+)?(route|trail|trails|tracking|recording)\b", text):
        return Intent.STOP_TRAIL
    if re.search(
        r"\b(take me back|navigate back|lead me back|guide me back|bring me back|go back( to (the )?(start|trail))?)\b",
        text,
    ):
        return Intent.NAVIGATE_BACK
    if re.search(r"\b(i reached the destination|destination reached|arrived|i am here|i'm here)\b", text):
        return Intent.DESTINATION_REACHED
    if re.search(r"\b(choose left|go left|take left|left option)\b", text):
        return Intent.CHOOSE_LEFT
    if re.search(r"\b(choose right|go right|take right|right option)\b", text):
        return Intent.CHOOSE_RIGHT
    if re.search(r"\b(choose saved route|saved route|use saved route|take( the)? saved (path|route)|take the recorded route|go the way i came|go back the same way)\b", text):
        return Intent.CHOOSE_SAVED_ROUTE
    if re.search(r"\b(choose alternate route|alternate route|use alternate route)\b", text):
        return Intent.CHOOSE_ALTERNATE_ROUTE
    return None


def _normalize_trail_text(text):
    text = (text or "").lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_explicit_stop_listening(text):
    normalized = _normalize_trail_text(text)
    return normalized in {"stop", "stop listening", "be quiet", "cancel", "thats enough", "that's enough", "no more"}


def classify_intent_fallback(text, detected_kind=None, last_question_type="none"):
    text = f" {text.lower()} "
    trail_intent = is_trail_command(text)
    if trail_intent is not None:
        return trail_intent
    permission_reply = infer_permission_reply(text)
    if last_question_type in {"initial_permission", "follow_up_offer"} and permission_reply == "yes":
        if last_question_type == "follow_up_offer" and detected_kind == "plant":
            return Intent.MORE_DETAIL
        return confirmation_action(detected_kind)
    if last_question_type == "more_detail_offer" and permission_reply == "yes":
        return Intent.MORE_DETAIL
    if last_question_type == "retry_clearer_view" and permission_reply == "yes":
        return Intent.IDENTIFY_CURRENT_OBJECT
    if permission_reply == "no":
        return Intent.CANCEL
    if permission_reply == "stop":
        trail_intent = is_trail_command(text)
        if trail_intent is not None:
            return trail_intent
        return Intent.STOP_LISTENING if is_explicit_stop_listening(text) else Intent.ASK_CLARIFICATION
    if permission_reply == "repeat":
        return Intent.REPEAT_LAST_MESSAGE
    if is_retry_request(text):
        return Intent.IDENTIFY_CURRENT_OBJECT
    if has(text, r"\b(stop|that'?s enough|no more|be quiet|cancel)\b"):
        return Intent.STOP_LISTENING if is_explicit_stop_listening(text) else Intent.ASK_CLARIFICATION
    if has(text, r"\b(no|nope|not now|leave it|i'?m good|i am good|don't|do not)\b"):
        return Intent.CANCEL
    if has(text, r"\b(repeat|again|say that|one more time)\b"):
        return Intent.REPEAT_LAST_MESSAGE
    if has(text, r"\b(tell me more|more|give me more information|yes tell me more|what else|explain more)\b"):
        return Intent.MORE_DETAIL
    if has(text, r"\b(help me|can you help me|describe this|describe it|what am i looking at|what is this|what is in front of me|what do you see)\b"):
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


def _parse_gemini_intent_response(text):
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        intent_name = str(data.get("intent") or data.get("action") or "").strip()
        confidence = float(data.get("confidence", 0))
        if intent_name not in GEMINI_INTENTS:
            return None, confidence
        return Intent(intent_name), confidence
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return None, 0.0


def confirm_ambiguous_intent_with_gemini(text, config, detected_kind=None, last_message="", last_question_type="none", interaction_context=None):
    from .interaction_pause import interaction_is_paused

    if interaction_is_paused():
        print("GEMINI_INTENT_CONFIRMATION result=ASK_CLARIFICATION reason=interaction_paused")
        return None
    context = interaction_context or {}
    print(f'GEMINI_INTENT_CONFIRMATION started transcript="{text}"')
    if not getattr(config, "openrouter_api_key", ""):
        print("GEMINI_INTENT_CONFIRMATION result=ASK_CLARIFICATION reason=missing_api_key")
        return None

    prompt = f"""
Return JSON only. No markdown or explanation.
Choose one intent from this exact list: {", ".join(sorted(GEMINI_INTENTS))}, ASK_CLARIFICATION.
Transcript: {text!r}
Interaction type: {context.get("interaction_type", "main_command")}
Visual interaction active: {bool(context.get("visual_interaction_active", False))}
Visual interaction type: {context.get("visual_interaction_type") or detected_kind or "none"}
Trail recording active: {bool(context.get("trail_recording_active", False))}
Assistant prompt: {last_message!r}
Last assistant question type: {last_question_type!r}

Interpret incomplete speech only when the intent is clear from this context. Otherwise choose ASK_CLARIFICATION.
Schema: {{"intent":"START_TRAIL","confidence":0.94,"normalized_command":"start the trail"}}
"""
    try:
        response = generate_openrouter(prompt, config)
    except Exception:
        print("GEMINI_INTENT_CONFIRMATION result=ASK_CLARIFICATION reason=request_failed")
        return None
    intent, confidence = _parse_gemini_intent_response(response)
    if intent is not None and confidence >= GEMINI_INTENT_CONFIDENCE:
        print(f"GEMINI_INTENT_CONFIRMATION result={intent.value} confidence={confidence:.2f}")
        return intent
    reason = "invalid_response" if not response or intent is None else "low_confidence"
    print(f"GEMINI_INTENT_CONFIRMATION result=ASK_CLARIFICATION reason={reason}")
    return None


def classify_intent_with_source(text, config=None, detected_kind=None, last_message="", ocr_text="", clip_confidence=None, last_question_type="none", interaction_context=None):
    permission_reply = infer_permission_reply(text or "")
    if text and last_question_type in {"initial_permission", "follow_up_offer"} and permission_reply == "yes":
        if last_question_type == "follow_up_offer" and detected_kind == "plant":
            return Intent.MORE_DETAIL, "permission_rule"
        return confirmation_action(detected_kind), "permission_rule"
    if text and last_question_type == "more_detail_offer" and permission_reply == "yes":
        return Intent.MORE_DETAIL, "permission_rule"
    if text and last_question_type == "retry_clearer_view" and permission_reply == "yes":
        return Intent.IDENTIFY_CURRENT_OBJECT, "permission_rule"
    if text and permission_reply == "no":
        return Intent.CANCEL, "permission_rule"
    if text and permission_reply == "stop":
        trail_intent = is_trail_command(text)
        if trail_intent is not None:
            return trail_intent, "trail_rule"
        if is_explicit_stop_listening(text):
            return Intent.STOP_LISTENING, "permission_rule"
        return Intent.ASK_CLARIFICATION, "permission_rule"
    if text and permission_reply == "repeat":
        return Intent.REPEAT_LAST_MESSAGE, "permission_rule"
    if text and is_retry_request(text):
        return Intent.IDENTIFY_CURRENT_OBJECT, "rule"
    if text and has(text, r"\b(what plant is this|identify this plant|tell me about this plant)\b"):
        return Intent.IDENTIFY_PLANT, "rule"
    if text and has(text, r"\b(help me|can you help me|can you tell me what this is|can you tell me what that is|what is this|what is that|what is in front of me|describe what i am looking at|what do you see)\b"):
        return Intent.WHAT_AM_I_LOOKING_AT, "rule"
    if text and is_acknowledgement(text):
        return Intent.CANCEL, "rule"
    trail_intent = is_trail_command(text)
    if trail_intent is not None:
        return trail_intent, "trail_rule"
    if text and is_general_question(text):
        return Intent.GENERAL_QUESTION, "rule"
    if text and last_question_type in {"initial_permission", "follow_up_offer"} and detected_kind in {"plant", "sign"} and is_help_prompt(last_message) and is_confirmation_reply(text):
        if last_question_type == "follow_up_offer" and detected_kind == "plant":
            return Intent.MORE_DETAIL, "rule"
        return confirmation_action(detected_kind), "rule"
    if text and last_question_type == "more_detail_offer" and is_confirmation_reply(text):
        return Intent.MORE_DETAIL, "rule"
    if text and last_question_type == "retry_clearer_view" and is_confirmation_reply(text):
        return Intent.IDENTIFY_CURRENT_OBJECT, "rule"
    if text and last_question_type in {"initial_permission", "follow_up_offer"} and is_confirmation_reply(text):
        if last_question_type == "follow_up_offer" and detected_kind == "plant":
            return Intent.MORE_DETAIL, "rule"
        return confirmation_action(detected_kind), "rule"
    fallback_intent = classify_intent_fallback(text or "", detected_kind, last_question_type)
    if fallback_intent != Intent.ASK_CLARIFICATION:
        return fallback_intent, "fallback"

    confirmed_intent = confirm_ambiguous_intent_with_gemini(
        text or "",
        config,
        detected_kind,
        last_message,
        last_question_type,
        interaction_context,
    )
    if confirmed_intent is not None:
        return confirmed_intent, "gemini_intent_confirmation"
    return Intent.ASK_CLARIFICATION, "fallback"


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
    assert classify_intent_fallback("yea can u", "sign", "initial_permission") == Intent.EXPLAIN_SIGN_MEANING
    assert classify_intent_fallback("would be great", "sign", "initial_permission") == Intent.EXPLAIN_SIGN_MEANING
    assert classify_intent_with_source("would be great", None, "sign", "I may be seeing a sign here. Would you like help with it?", last_question_type="initial_permission")[0] == Intent.EXPLAIN_SIGN_MEANING
    assert classify_intent_with_source("that sounds useful", None, "sign", "Would you like help with it?", last_question_type="initial_permission") == (Intent.EXPLAIN_SIGN_MEANING, "permission_rule")
    assert classify_intent_with_source("I would appreciate that", None, "plant", "Should I identify it?", last_question_type="initial_permission") == (Intent.EXPLAIN_PLANT, "permission_rule")
    assert classify_intent_with_source("maybe later", None, "sign", "Would you like help with it?", last_question_type="initial_permission") == (Intent.CANCEL, "permission_rule")
    assert classify_intent_fallback("why not", "plant", "follow_up_offer") == Intent.MORE_DETAIL
    assert classify_intent_fallback("why not", "sign", "follow_up_offer") == Intent.EXPLAIN_SIGN_MEANING
    assert classify_intent_fallback("no not now", "plant") == Intent.CANCEL
    assert classify_intent_fallback("Repeat that", "sign") == Intent.REPEAT_LAST_MESSAGE
    assert classify_intent_fallback("tell me more", "plant") == Intent.MORE_DETAIL
    assert classify_intent_fallback("what am I looking at", "other") == Intent.WHAT_AM_I_LOOKING_AT
    assert classify_intent_fallback("oh can you help me", "other") == Intent.WHAT_AM_I_LOOKING_AT
    assert classify_intent_with_source("oh can you help me", None, "other")[0] == Intent.WHAT_AM_I_LOOKING_AT
    assert classify_intent_fallback("what plant is this", "other") == Intent.IDENTIFY_PLANT
    assert classify_intent_fallback("what is photosynthesis", "other") == Intent.GENERAL_QUESTION
    assert classify_intent_fallback("stop", "sign") == Intent.STOP_LISTENING
    assert classify_intent_fallback("try again", "plant") == Intent.IDENTIFY_CURRENT_OBJECT
    assert confirmation_action("plant") == Intent.EXPLAIN_PLANT
    assert is_trail_command("start the trail") == Intent.START_TRAIL
    assert is_trail_command("take me back") == Intent.NAVIGATE_BACK
    assert is_trail_command("choose left") == Intent.CHOOSE_LEFT
    assert is_trail_command("I reached the destination") == Intent.DESTINATION_REACHED


if __name__ == "__main__":
    _demo()
