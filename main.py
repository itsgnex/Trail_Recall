import argparse
from datetime import datetime
from dataclasses import replace
from pathlib import Path
import queue
import threading
import time
import cv2
import numpy as np

from src import app_log
from src.camera import Camera, split_stream_urls, wait_for_publisher
from src.config import Config
from src.latency import log_stage, new_interaction, set_interaction
from src.intent import Intent, classify_intent_with_source
from src.interaction_pause import interaction_is_paused, log_interaction_skip
from src.clip_classifier import get_crop_embedding
from src.llm import answer_for, answer_general_question_with_1b, describe_current_object, generate_more_detail_response
from src.ocr import read_text
from src.phrases import (
    get_cancel_response,
    get_clarification_response,
    get_follow_up_offer,
    get_follow_up_missed_response,
    get_prompt,
    get_repeat_response,
    get_unclear_plant_response,
    get_unclear_sign_response,
    get_wake_check_response,
    format_remembered_answer,
    get_trail_started_response,
    get_trail_saved_response,
    get_trail_navigate_response,
    get_trail_failed_response,
)
from src.audio_http import TtsHttpServer, update_runtime_status
from src.config import DEFAULT_MENTRA_RTSP_URL
from src.mic_ingest import MicIngestServer
from src.network_diag import print_network_check
from src.rtmp_audio_ingest import RtmpAudioIngest
from src.session_state import SessionState
from src.scene_memory import lookup_scene_memory, restore_plant_id, store_scene_memory
from src.sign_memory import add_or_update_sign_memory, cleanup_old_signs, find_similar_sign, is_recent_duplicate
from src.speech_in import calibrate_wake_ambient, is_incomplete_command, is_wake_check_command, listen, listen_for_wake_phrase, preload_whisper_model, strip_wake_phrase
from src.speech_out import cancel_reply_capture, is_speaking, log_first_transcript_after_trail, speak
from src.trail_phone import android_bridge_diagnostics, check_trail_health, format_trail_error, send_trail_command
from src.trigger import DwellTrigger
from src.vision import analyze_crop, draw_focus_box, focus_crop
from src.vision_llm import verify_same_sign


def save_debug_crop(kind, crop):
    path = Path("debug_crops") / f"{kind}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.jpg"
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), crop)
    print(f"debug crop saved: {path}")
    return str(path)


def capture_center_crop(camera, config):
    frame = camera.read()
    if frame is None:
        return None
    crop, _ = focus_crop(frame, config.focus_fraction)
    return crop


def maybe_store_scene_memory(kind, crop, answer, config, state, crop_path=None):
    if not getattr(config, "scene_memory_enabled", True):
        return
    if not answer or needs_clearer_view(answer):
        return
    store_scene_memory(
        kind,
        "scene",
        crop,
        answer,
        crop_path=crop_path or state.last_crop_path,
        vision_result=state.last_vision_result,
        plant_id_result=state.last_plant_id_result,
        ttl_seconds=getattr(config, "scene_memory_ttl_seconds", 3600),
        max_items=getattr(config, "scene_memory_max_items", 120),
    )


def crop_signature(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return small


def crop_changed_enough(current, previous, threshold=5.0):
    if previous is None:
        return True
    return float(np.mean(np.abs(current.astype(np.int16) - previous.astype(np.int16)))) >= threshold


VISUAL_IDLE = "IDLE"
VISUAL_PROMPT_PLAYING = "PROMPT_PLAYING"
VISUAL_WAITING_FOR_REPLY = "WAITING_FOR_REPLY"
VISUAL_ANALYZING = "ANALYZING"
VISUAL_RESULT_PLAYING = "RESULT_PLAYING"
VISUAL_COOLDOWN = "COOLDOWN"


def _visual_signature_distance(left, right):
    if left is None or right is None:
        return 999.0
    return float(np.mean(np.abs(left.astype(np.int16) - right.astype(np.int16))))


def _visual_same_object(kind, signature, state, threshold=8.0):
    if not state.visual_object_present or state.visual_object_kind != kind:
        return False
    return True


def _visual_set_state(state, value):
    previous = state.visual_prompt_state
    state.visual_prompt_state = value
    if previous != value:
        app_log.debug(f"VISUAL_STATE from={previous} to={value}")


def _visual_log_suppressed(reason, **values):
    suffix = " ".join(f"{key}={value}" for key, value in values.items() if value not in {None, ""})
    app_log.rate_limited(f"visual_suppressed_{reason}", 10, f"VISUAL_PROMPT suppressed reason={reason}" + (f" {suffix}" if suffix else ""), level="DEBUG")


def visual_mark_absence(kind, config, state, crop=None, now=None):
    now = time.monotonic() if now is None else now
    if state.visual_prompt_state == VISUAL_COOLDOWN and now >= state.visual_cooldown_until and not state.visual_object_present:
        _visual_set_state(state, VISUAL_IDLE)
    if kind in {"plant", "sign"} and kind == state.visual_object_kind:
        state.visual_object_absent_since = 0.0
        state.visual_object_absent_frames = 0
        state.visual_object_last_seen = now
        return
    if not state.visual_object_present:
        return
    if not state.visual_object_absent_since:
        state.visual_object_absent_since = now
        state.visual_object_absent_signature = crop_signature(crop) if crop is not None else None
    state.visual_object_absent_frames += 1
    current_signature = crop_signature(crop) if crop is not None else state.visual_object_absent_signature
    scene_changed = (
        current_signature is not None
        and _visual_signature_distance(current_signature, state.visual_object_signature)
        >= getattr(config, "visual_rearm_scene_change_threshold", 18.0)
    )
    if (
        now - state.visual_object_absent_since >= getattr(config, "visual_object_gone_seconds", 3.0)
        and state.visual_object_absent_frames >= getattr(config, "visual_object_gone_min_frames", 8)
        and scene_changed
    ):
        state.visual_object_present = False
        state.visual_object_absent_since = 0.0
        state.visual_object_absent_frames = 0
        state.visual_object_absent_signature = None
        if state.visual_prompt_state == VISUAL_COOLDOWN and now >= state.visual_cooldown_until:
            _visual_set_state(state, VISUAL_IDLE)


def visual_prompt_should_start(kind, crop, config, state, now=None):
    now = time.monotonic() if now is None else now
    if kind not in {"plant", "sign"}:
        visual_mark_absence(kind, config, state, crop=crop, now=now)
        return False
    if interaction_is_paused():
        log_interaction_skip("VISUAL_PROMPT_SKIPPED")
        return False
    if state.visual_prompt_state == VISUAL_COOLDOWN and now >= state.visual_cooldown_until and not state.visual_object_present:
        _visual_set_state(state, VISUAL_IDLE)
    if is_speaking():
        _visual_log_suppressed("tts_playing")
        return False
    if state.visual_prompt_state != VISUAL_IDLE:
        reason = "waiting_for_reply" if state.visual_prompt_state == VISUAL_WAITING_FOR_REPLY else state.visual_prompt_state.lower()
        _visual_log_suppressed(reason)
        return False
    if now < state.visual_cooldown_until:
        _visual_set_state(state, VISUAL_COOLDOWN)
        _visual_log_suppressed("cooldown", remainingSeconds=int(state.visual_cooldown_until - now))
        return False

    signature = crop_signature(crop)
    if _visual_same_object(kind, signature, state):
        state.visual_object_last_seen = now
        state.visual_object_absent_since = 0.0
        _visual_log_suppressed("same_object")
        return False

    state.visual_interaction_count += 1
    state.visual_interaction_id = f"{kind}-{state.visual_interaction_count}"
    state.visual_object_kind = kind
    state.visual_object_signature = signature
    state.visual_object_present = True
    state.visual_object_last_seen = now
    state.visual_object_absent_since = 0.0
    state.visual_object_absent_frames = 0
    state.visual_object_absent_signature = None
    state.visual_reply_retries = 0
    _visual_set_state(state, VISUAL_PROMPT_PLAYING)
    print(f"VISUAL_PROMPT started type={kind} visualId={state.visual_interaction_id}")
    return True


def visual_interaction_complete(state, config, result, now=None):
    now = time.monotonic() if now is None else now
    state.visual_cooldown_until = now + getattr(config, "visual_prompt_cooldown_seconds", 7.0)
    previous = state.visual_prompt_state
    _visual_set_state(state, VISUAL_COOLDOWN)
    state.visual_reply_retries = 0
    if previous == VISUAL_RESULT_PLAYING:
        print("VISUAL_STATE from=RESULT_PLAYING to=COOLDOWN")
    print(f"VISUAL_INTERACTION completed result={result}")


def visual_speak_final(text, state, config, result="yes"):
    _visual_set_state(state, VISUAL_RESULT_PLAYING)

    def _complete(_event_id, _reason):
        visual_interaction_complete(state, config, result)

    return speak(text, expect_reply=False, on_complete=_complete)


def needs_clearer_view(answer):
    text = (answer or "").lower()
    return any(
        phrase in text
        for phrase in (
            "not clear enough",
            "cannot read",
            "can't read",
            "hard to see",
            "too blurry",
            "too far",
            "closer",
            "steadier",
            "straighter",
            "clearer view",
            "centering",
            "center it",
        )
    )


def answer_asks_follow_up(answer):
    text = (answer or "").strip().lower()
    if not text.endswith("?"):
        return False
    return any(
        phrase in text
        for phrase in (
            "would you like",
            "do you want",
            "should i",
            "can i",
            "want me to",
            "if you want",
        )
    )


def record_sign_memory(crop, crop_path, config, state, now=None):
    if not getattr(config, "sign_memory_enabled", True):
        return None
    embedding = get_crop_embedding(crop)
    if embedding is None:
        return None
    analysis = state.last_vision_result or {}
    return add_or_update_sign_memory(
        crop_path=crop_path,
        embedding=embedding,
        visible_text=(analysis.get("visible_text") or ""),
        symbol_or_icon=(analysis.get("symbol_or_icon") or ""),
        plain_meaning=(analysis.get("plain_meaning") or ""),
        description=(analysis.get("description") or ""),
        final_answer=(state.last_core_answer or state.last_final_answer or ""),
        last_prompted_time=now or time.monotonic(),
        last_explained_time=now or time.monotonic(),
        max_items=getattr(config, "sign_memory_max_items", 30),
        ttl_seconds=getattr(config, "sign_memory_ttl_seconds", 600),
        duplicate_threshold=getattr(config, "sign_clip_possible_duplicate_threshold", 0.82),
        now=now,
    )


def maybe_suppress_sign_prompt(crop, crop_path, config, state):
    if not getattr(config, "sign_memory_enabled", True):
        return False, None, None, None
    embedding = get_crop_embedding(crop)
    if embedding is None:
        return False, None, None, None

    now = time.monotonic()
    cleanup_old_signs(now, getattr(config, "sign_memory_ttl_seconds", 600))
    match, similarity = find_similar_sign(
        embedding,
        min_similarity=getattr(config, "sign_clip_possible_duplicate_threshold", 0.82),
        now=now,
        ttl_seconds=getattr(config, "sign_memory_ttl_seconds", 600),
    )
    if match is None:
        print("sign memory: no similar sign found")
        return False, None, None, None

    print(f"sign memory: possible duplicate similarity={similarity:.2f}")
    suppress_seconds = getattr(config, "sign_duplicate_suppress_seconds", 180)
    if is_recent_duplicate(match, now, suppress_seconds):
        cached = match.get("final_answer")
        print("sign memory: recent duplicate, suppressing prompt")
        return True, cached, embedding, match

    if similarity < getattr(config, "sign_clip_duplicate_threshold", 0.88):
        return False, None, embedding, None

    if getattr(config, "sign_gemma_verify_duplicates", True):
        verdict = verify_same_sign(crop_path, match, config)
        if bool(verdict.get("same_sign")):
            print("sign memory: same sign confirmed, suppressing prompt")
            add_or_update_sign_memory(
                crop_path=crop_path,
                embedding=embedding,
                max_items=getattr(config, "sign_memory_max_items", 30),
                ttl_seconds=getattr(config, "sign_memory_ttl_seconds", 600),
                duplicate_threshold=getattr(config, "sign_clip_possible_duplicate_threshold", 0.82),
                now=now,
            )
            return True, match.get("final_answer") or None, embedding, match
        print("sign memory: different sign, continuing")
        return False, None, embedding, None

    print("sign memory: duplicate threshold reached, continuing without verify")
    return False, None, embedding, None


def listen_for_follow_up_reply(prompt, config, state, detected_kind, ocr_text="", record_seconds=None):
    if interaction_is_paused():
        log_interaction_skip("FOLLOW_UP_SKIPPED")
        return "", None, None
    state.last_assistant_question_type = "follow_up_offer"
    state.follow_up_enabled = True
    state.awaiting_follow_up_reply = True
    try:
        heard = listen(
            config,
            typed_fallback=False,
            record_seconds=record_seconds or getattr(config, "follow_up_timeout_seconds", 1.5),
            label="recording follow-up",
            after_tts=True,
            silence_ms=getattr(config, "follow_up_silence_ms", 300),
        )
        if not heard:
            return "", None, None

        log_stage("INTENT_ROUTING_START")
        intent, source = classify_intent_with_source(
            heard,
            config,
            detected_kind,
            prompt,
            ocr_text,
            last_question_type=state.last_assistant_question_type,
            interaction_context=_intent_context(state, "follow_up"),
        )
        log_stage("INTENT_ROUTING_END", intent=intent.value, source=source)
        print(f'follow-up said: "{heard}" -> {intent.value} via {source}')
        state.last_user_transcript = heard
        state.last_action = intent.value
        return heard, intent, source
    finally:
        state.follow_up_enabled = False
        state.awaiting_follow_up_reply = False


def _intent_context(state, interaction_type):
    visual_active = state.visual_prompt_state not in {VISUAL_IDLE, VISUAL_COOLDOWN}
    return {
        "interaction_type": interaction_type,
        "visual_interaction_active": visual_active,
        "visual_interaction_type": state.last_detected_kind if visual_active else None,
        "trail_recording_active": state.trail_recording_active,
    }


def _clarification_for_context(state, transcript=""):
    if state.visual_prompt_state not in {VISUAL_IDLE, VISUAL_COOLDOWN}:
        if state.last_detected_kind == "sign":
            return "Would you like me to read the sign?"
        if state.last_detected_kind == "plant":
            return "Would you like me to identify the plant?"
    if state.trail_recording_active or any(word in (transcript or "").lower() for word in ("trail", "route", "back", "record", "start")):
        return "Would you like me to start the trail or take you back?"
    return get_clarification_response()


def handle_trail_intent(intent, config, state=None):
    actions = {
        Intent.START_TRAIL: "start",
        Intent.STOP_TRAIL: "stop",
        Intent.NAVIGATE_BACK: "navigate-back",
        Intent.DESTINATION_REACHED: "destination-reached",
        Intent.CHOOSE_LEFT: "choose-left",
        Intent.CHOOSE_RIGHT: "choose-right",
        Intent.CHOOSE_SAVED_ROUTE: "choose-saved-route",
        Intent.CHOOSE_ALTERNATE_ROUTE: "choose-alternate-route",
    }
    action = actions.get(intent)
    if action:
        ok, detail = send_trail_command(action, config)
        trail_confirmations = {
            Intent.START_TRAIL: get_trail_started_response,
            Intent.STOP_TRAIL: get_trail_saved_response,
            Intent.NAVIGATE_BACK: get_trail_navigate_response,
        }
        confirmation = trail_confirmations.get(intent)
        if ok and confirmation:
            if state is not None and intent == Intent.START_TRAIL:
                state.trail_recording_active = True
            elif state is not None and intent == Intent.STOP_TRAIL:
                state.trail_recording_active = False
            speak(confirmation(), trail_command=True)
        elif not ok:
            speak(format_trail_error(detail), trail_command=intent in trail_confirmations)
        return True
    return False


def handle_voice_command(transcript, config, state, camera):
    if interaction_is_paused():
        log_interaction_skip("VOICE_COMMAND_SKIPPED")
        return
    state.is_busy = True
    state.is_processing_command = True
    try:
        command = strip_wake_phrase(
            transcript,
            getattr(config, "wake_phrases", ()),
            getattr(config, "allow_single_word_wake", False),
        )
        if not command:
            speak("Yes?")
            command = listen(
                config,
                typed_fallback=False,
                record_seconds=getattr(config, "command_record_seconds", 2.5),
                label="recording command",
                after_tts=True,
            )
        command = strip_wake_phrase(
            command,
            getattr(config, "wake_phrases", ()),
            getattr(config, "allow_single_word_wake", False),
        )
        command = (command or "").strip()
        if is_wake_check_command(command):
            speak(get_wake_check_response())
            return
        if is_incomplete_command(command):
            speak("I did not catch the full question. Please say it again.")
            command = listen(
                config,
                typed_fallback=False,
                record_seconds=getattr(config, "command_record_seconds", 2.5),
                label="recording command",
                after_tts=True,
            )
            command = strip_wake_phrase(
                command,
                getattr(config, "wake_phrases", ()),
                getattr(config, "allow_single_word_wake", False),
            )
        command = (command or "").strip()
        if is_wake_check_command(command):
            speak(get_wake_check_response())
            return
        print(f'COMMAND transcript="{command}"')
        if not command or is_incomplete_command(command):
            speak(get_wake_check_response())
            return

        log_stage("INTENT_ROUTING_START")
        intent, source = classify_intent_with_source(
            command,
            config,
            state.last_detected_kind,
            state.last_prompt or "",
            last_question_type=state.last_assistant_question_type,
            interaction_context=_intent_context(state, "wake_command"),
        )
        log_stage("INTENT_ROUTING_END", intent=intent.value, source=source)
        print(f"INTENT type={intent.value} source={source}")
        state.last_user_transcript = command
        state.last_action = intent.value

        if intent in {Intent.CANCEL, Intent.STOP_LISTENING}:
            speak(get_cancel_response())
            state.last_assistant_question_type = "none"
            return
        if intent == Intent.REPEAT_LAST_MESSAGE:
            speak(get_repeat_response())
            if state.last_core_answer:
                speak(state.last_core_answer)
            return
        if intent == Intent.SPEAK_SLOWER:
            speak("Of course. I’ll keep it slower and brief.")
            return
        if handle_trail_intent(intent, config, state):
            return
        if intent == Intent.MORE_DETAIL:
            answer = generate_more_detail_response(state, config)
            state.last_core_answer = answer
            state.last_final_answer = answer
            speak(answer)
            return
        if intent == Intent.GENERAL_QUESTION:
            answer = answer_general_question_with_1b(command, config)
            state.last_core_answer = answer
            state.last_final_answer = answer
            speak(answer)
            return

        if intent in {Intent.WHAT_AM_I_LOOKING_AT, Intent.DESCRIBE_CURRENT_OBJECT, Intent.IDENTIFY_CURRENT_OBJECT, Intent.READ_SIGN_TEXT, Intent.EXPLAIN_SIGN_MEANING, Intent.IDENTIFY_PLANT, Intent.EXPLAIN_PLANT}:
            crop = capture_center_crop(camera, config)
            if crop is None:
                print("camera frame was unavailable for voice command")
                return
            crop_kind = "sign" if intent in {Intent.READ_SIGN_TEXT, Intent.EXPLAIN_SIGN_MEANING} else "plant" if intent in {Intent.IDENTIFY_PLANT, Intent.EXPLAIN_PLANT} else state.last_detected_kind or "object"
            crop_path = save_debug_crop(crop_kind, crop)
            log_stage("VISION_REQUEST_START", kind=crop_kind)
            if crop_kind == "sign":
                state.last_detected_kind = "sign"
                answer = answer_for("sign", crop, intent, config, state)
            elif crop_kind == "plant":
                state.last_detected_kind = "plant"
                answer = answer_for("plant", crop, intent, config, state)
            else:
                answer = describe_current_object(crop, config, state)
            log_stage("VISION_REQUEST_END", kind=crop_kind)

            state.last_crop_path = crop_path
            state.last_core_answer = answer
            state.last_final_answer = answer
            state.last_answer_time = time.monotonic()
            speak(answer)
            if state.last_detected_kind == "sign" or crop_kind == "sign":
                record_sign_memory(crop, crop_path, config, state, now=state.last_answer_time)
            maybe_store_scene_memory(crop_kind, crop, answer, config, state, crop_path)
            return

        clarification_prompt = _clarification_for_context(state, command)
        state.awaiting_follow_up_reply = True
        speak(clarification_prompt)
        heard, follow_up_intent, follow_up_source = listen_for_follow_up_reply(
            clarification_prompt,
            config,
            state,
            state.last_detected_kind,
            state.last_prompt or "",
            record_seconds=getattr(config, "follow_up_timeout_seconds", 1.5),
        )
        if follow_up_intent is None:
            state.last_assistant_question_type = "none"
            state.awaiting_follow_up_reply = False
            return
        if follow_up_intent == Intent.ASK_CLARIFICATION:
            speak(get_follow_up_missed_response())
            state.last_assistant_question_type = "none"
            state.awaiting_follow_up_reply = False
            return
        state.last_assistant_question_type = "none"
        intent = follow_up_intent
        source = follow_up_source
        log_stage("INTENT_ROUTING_END", intent=intent.value, source=source)
        state.last_user_transcript = heard
        state.last_action = intent.value
        if intent in {Intent.CANCEL, Intent.STOP_LISTENING}:
            speak(get_cancel_response())
            return
        if intent == Intent.REPEAT_LAST_MESSAGE:
            speak(get_repeat_response())
            if state.last_core_answer:
                speak(state.last_core_answer)
            return
        if intent == Intent.SPEAK_SLOWER:
            speak("Of course. I’ll keep it slower and brief.")
            return
        if handle_trail_intent(intent, config, state):
            return
        if intent == Intent.MORE_DETAIL:
            answer = generate_more_detail_response(state, config)
            state.last_core_answer = answer
            state.last_final_answer = answer
            speak(answer)
            return
        if intent == Intent.GENERAL_QUESTION:
            answer = answer_general_question_with_1b(heard, config)
            state.last_core_answer = answer
            state.last_final_answer = answer
            speak(answer)
            return
        if intent in {Intent.WHAT_AM_I_LOOKING_AT, Intent.DESCRIBE_CURRENT_OBJECT, Intent.IDENTIFY_CURRENT_OBJECT, Intent.READ_SIGN_TEXT, Intent.EXPLAIN_SIGN_MEANING, Intent.IDENTIFY_PLANT, Intent.EXPLAIN_PLANT}:
            crop = capture_center_crop(camera, config)
            if crop is None:
                print("camera frame was unavailable for follow-up voice command")
                return
            crop_kind = "sign" if intent in {Intent.READ_SIGN_TEXT, Intent.EXPLAIN_SIGN_MEANING} else "plant" if intent in {Intent.IDENTIFY_PLANT, Intent.EXPLAIN_PLANT} else state.last_detected_kind or "object"
            crop_path = save_debug_crop(crop_kind, crop)
            log_stage("VISION_REQUEST_START", kind=crop_kind)
            if crop_kind == "sign":
                state.last_detected_kind = "sign"
                answer = answer_for("sign", crop, intent, config, state)
            elif crop_kind == "plant":
                state.last_detected_kind = "plant"
                answer = answer_for("plant", crop, intent, config, state)
            else:
                answer = describe_current_object(crop, config, state)
            log_stage("VISION_REQUEST_END", kind=crop_kind)
            state.last_crop_path = crop_path
            state.last_core_answer = answer
            state.last_final_answer = answer
            state.last_answer_time = time.monotonic()
            speak(answer)
            if state.last_detected_kind == "sign" or crop_kind == "sign":
                record_sign_memory(crop, crop_path, config, state, now=state.last_answer_time)
            maybe_store_scene_memory(crop_kind, crop, answer, config, state, crop_path)
            return
        speak(_clarification_for_context(state, heard))
        return
    finally:
        state.is_processing_command = False
        state.is_busy = False


def wake_listener_loop(config, state, wake_queue, stop_event):
    if not (getattr(config, "voice_activation_mode", True) or getattr(config, "wake_mode", False)):
        return
    while not stop_event.is_set():
        if interaction_is_paused():
            log_interaction_skip("WAKE_SKIPPED")
            time.sleep(0.1)
            continue
        paused_for_tts = getattr(config, "pause_wake_during_tts", True) and is_speaking()
        if paused_for_tts or (getattr(config, "pause_wake_during_tts", True) and (getattr(state, "is_busy", False) or getattr(state, "is_processing_command", False) or getattr(state, "follow_up_enabled", False) or getattr(state, "awaiting_follow_up_reply", False))):
            time.sleep(0.1)
            continue
        wake_result = listen_for_wake_phrase(mic_index=getattr(config, "mic_device_index", 0), config=config, state=state)
        if wake_result:
            transcript = wake_result.get("transcript", wake_result) if isinstance(wake_result, dict) else wake_result
            if getattr(state, "is_busy", False):
                time.sleep(0.1)
                continue
            print("voice activation: speech captured")
            log_first_transcript_after_trail(transcript)
            wake_queue.put(wake_result)
            time.sleep(max(0.1, float(getattr(config, "wake_cooldown_seconds", 2))))
        else:
            time.sleep(0.05)


def handle_follow_up(crop, config, state, camera):
    if interaction_is_paused():
        log_interaction_skip("FOLLOW_UP_SKIPPED")
        return
    if not getattr(config, "follow_up_mode", True):
        return
    if state.last_assistant_question_type != "follow_up_offer":
        return

    state.follow_up_enabled = True
    state.follow_up_timeout_seconds = getattr(config, "follow_up_timeout_seconds", 1.5)
    try:
        for _ in range(getattr(config, "max_follow_up_turns", 2)):
            heard = listen(
                config,
                typed_fallback=False,
                record_seconds=state.follow_up_timeout_seconds,
                after_tts=True,
                silence_ms=getattr(config, "follow_up_silence_ms", 300),
            )
            if not heard:
                if getattr(config, "follow_up_silence_returns_to_scan", True):
                    state.last_assistant_question_type = "none"
                    return
                speak(get_follow_up_missed_response())
                state.last_assistant_question_type = "none"
                return

            intent, source = classify_intent_with_source(
                heard,
                config,
                state.last_detected_kind,
                state.last_final_answer or "",
                last_question_type=state.last_assistant_question_type,
                interaction_context=_intent_context(state, "follow_up"),
            )
            print(f'follow-up said: "{heard}" -> {intent.value} via {source}')
            state.last_user_transcript = heard
            previous_action = state.last_action

            if intent in {Intent.CANCEL, Intent.STOP_LISTENING}:
                speak(get_cancel_response())
                state.last_assistant_question_type = "none"
                return
            if intent == Intent.REPEAT_LAST_MESSAGE:
                speak(get_repeat_response())
                if state.last_core_answer:
                    speak(state.last_core_answer)
                continue
            if intent in {Intent.EXPLAIN_PLANT, Intent.EXPLAIN_SIGN_MEANING} and previous_action == intent.value:
                speak("I already checked this view. Try moving a little closer or centering it more clearly, and I can check again.")
                state.last_assistant_question_type = "retry_clearer_view"
                return
            if intent == Intent.MORE_DETAIL:
                answer = generate_more_detail_response(state, config)
            elif intent in {Intent.WHAT_AM_I_LOOKING_AT, Intent.DESCRIBE_CURRENT_OBJECT, Intent.IDENTIFY_CURRENT_OBJECT}:
                if intent == Intent.IDENTIFY_CURRENT_OBJECT:
                    state.last_assistant_question_type = "retry_clearer_view"
                answer = describe_current_object(crop, config, state)
            elif intent == Intent.ASK_CLARIFICATION:
                clarification_prompt = _clarification_for_context(state, heard)
                state.awaiting_follow_up_reply = True
                speak(clarification_prompt)
                heard, follow_up_intent, follow_up_source = listen_for_follow_up_reply(
                    clarification_prompt,
                    config,
                    state,
                    state.last_detected_kind,
                    state.last_final_answer or "",
                    record_seconds=getattr(config, "follow_up_timeout_seconds", 1.5),
                )
                if follow_up_intent is None:
                    state.last_assistant_question_type = "none"
                    return
                if follow_up_intent == Intent.ASK_CLARIFICATION:
                    speak(get_follow_up_missed_response())
                    state.last_assistant_question_type = "none"
                    return
                state.last_assistant_question_type = "none"
                intent = follow_up_intent
                source = follow_up_source
                state.last_user_transcript = heard
                state.last_action = intent.value
                if intent == Intent.MORE_DETAIL:
                    answer = generate_more_detail_response(state, config)
                elif intent == Intent.GENERAL_QUESTION:
                    answer = answer_general_question_with_1b(heard, config)
                elif intent in {Intent.WHAT_AM_I_LOOKING_AT, Intent.DESCRIBE_CURRENT_OBJECT, Intent.IDENTIFY_CURRENT_OBJECT}:
                    fresh_crop = capture_center_crop(camera, config)
                    if fresh_crop is not None:
                        crop = fresh_crop
                        state.last_crop_path = save_debug_crop(state.last_detected_kind or "object", crop)
                    if intent == Intent.IDENTIFY_CURRENT_OBJECT:
                        state.last_assistant_question_type = "retry_clearer_view"
                    answer = describe_current_object(crop, config, state)
                else:
                    answer = answer_for(state.last_detected_kind or "other", crop, intent, config, state)
            else:
                answer = answer_for(state.last_detected_kind or "other", crop, intent, config, state)

            state.last_action = intent.value
            state.last_core_answer = answer
            state.last_final_answer = answer
            state.last_answer_time = time.monotonic()
            speak(answer)
            if needs_clearer_view(answer):
                state.last_assistant_question_type = "retry_clearer_view"
                return
            maybe_store_scene_memory(state.last_detected_kind or "other", crop, answer, config, state)
            state.last_follow_up_offer = None
            if answer_asks_follow_up(answer):
                state.last_assistant_question_type = "follow_up_offer"
                continue
            elif getattr(config, "speak_follow_up_offer", False) and intent in {Intent.EXPLAIN_PLANT, Intent.EXPLAIN_SIGN_MEANING, Intent.WHAT_AM_I_LOOKING_AT, Intent.DESCRIBE_CURRENT_OBJECT, Intent.IDENTIFY_CURRENT_OBJECT}:
                state.last_follow_up_offer = get_follow_up_offer()
                speak(state.last_follow_up_offer)
                state.last_assistant_question_type = "follow_up_offer"
                continue
            else:
                state.last_assistant_question_type = "none"
                return
    finally:
        state.follow_up_enabled = False


def handle_trigger(kind, crop, config, state, camera, crop_path=None):
    if interaction_is_paused():
        log_interaction_skip("VISUAL_INTERACTION_SKIPPED")
        return
    state.is_busy = True
    try:
        crop_path = crop_path or save_debug_crop(kind, crop)
        if kind == "plant":
            prompt = get_prompt("plant")
        elif kind == "sign":
            prompt = get_prompt("sign")
        else:
            prompt = get_prompt("unclear")

        state.last_detected_kind = kind
        state.last_crop_path = crop_path
        state.last_prompt = prompt
        state.last_assistant_question_type = "initial_permission"
        _visual_set_state(state, VISUAL_PROMPT_PLAYING)
        speak(prompt, expect_reply=True)
        _visual_set_state(state, VISUAL_WAITING_FOR_REPLY)
        state.awaiting_follow_up_reply = True
        try:
            heard = listen(
                config,
                record_seconds=getattr(config, "confirmation_record_seconds", 2),
                after_tts=True,
                silence_ms=getattr(config, "confirmation_silence_ms", 350),
            )
        finally:
            state.awaiting_follow_up_reply = False
        if not heard:
            state.last_assistant_question_type = "none"
            visual_interaction_complete(state, config, "timeout")
            return
        ocr_text = read_text(crop) if kind == "sign" else ""
        log_stage("INTENT_ROUTING_START")
        intent, source = classify_intent_with_source(
            heard,
            config,
            kind,
            prompt,
            ocr_text,
            last_question_type=state.last_assistant_question_type,
            interaction_context=_intent_context(state, "reply"),
        )
        log_stage("INTENT_ROUTING_END", intent=intent.value, source=source)
        print(f'user said: "{heard or "[nothing heard]"}" -> {intent.value} via {source}')
        state.last_user_transcript = heard
        state.last_action = intent.value

        if intent == Intent.REPEAT_LAST_MESSAGE:
            _visual_set_state(state, VISUAL_PROMPT_PLAYING)
            speak(get_repeat_response(), expect_reply=False)
            speak(prompt, expect_reply=True)
            _visual_set_state(state, VISUAL_WAITING_FOR_REPLY)
            heard = listen(
                config,
                record_seconds=getattr(config, "confirmation_record_seconds", 2),
                after_tts=True,
                silence_ms=getattr(config, "confirmation_silence_ms", 350),
            )
            log_stage("INTENT_ROUTING_START")
            intent, source = classify_intent_with_source(
                heard,
                config,
                kind,
                prompt,
                ocr_text,
                last_question_type=state.last_assistant_question_type,
                interaction_context=_intent_context(state, "reply"),
            )
            log_stage("INTENT_ROUTING_END", intent=intent.value, source=source)
            print(f'user said: "{heard or "[nothing heard]"}" -> {intent.value} via {source}')
            state.last_user_transcript = heard
            state.last_action = intent.value

        if intent in {Intent.CANCEL, Intent.STOP_LISTENING}:
            _visual_set_state(state, VISUAL_RESULT_PLAYING)
            visual_speak_final("Okay.", state, config, "no")
            state.last_assistant_question_type = "none"
            return
        if intent == Intent.ASK_CLARIFICATION:
            if state.visual_reply_retries >= getattr(config, "visual_reply_max_retries", 1):
                state.last_assistant_question_type = "none"
                visual_interaction_complete(state, config, "clarification_failed")
                return
            state.visual_reply_retries += 1
            clarification_prompt = _clarification_for_context(state, heard)
            state.awaiting_follow_up_reply = True
            _visual_set_state(state, VISUAL_PROMPT_PLAYING)
            speak(clarification_prompt, expect_reply=True)
            _visual_set_state(state, VISUAL_WAITING_FOR_REPLY)
            heard, follow_up_intent, follow_up_source = listen_for_follow_up_reply(
                clarification_prompt,
                config,
                state,
                kind,
                ocr_text,
                record_seconds=getattr(config, "follow_up_timeout_seconds", 1.5),
            )
            if follow_up_intent is None:
                state.last_assistant_question_type = "none"
                visual_interaction_complete(state, config, "timeout")
                return
            if follow_up_intent == Intent.ASK_CLARIFICATION:
                state.last_assistant_question_type = "none"
                visual_interaction_complete(state, config, "clarification_failed")
                return
            state.last_assistant_question_type = "none"
            intent = follow_up_intent
            source = follow_up_source
        if intent == Intent.SPEAK_SLOWER:
            visual_speak_final("Of course. I’ll keep it slower and brief.", state, config, "yes")
            state.last_assistant_question_type = "none"
            return

        _visual_set_state(state, VISUAL_ANALYZING)
        if intent in {Intent.WHAT_AM_I_LOOKING_AT, Intent.DESCRIBE_CURRENT_OBJECT, Intent.IDENTIFY_CURRENT_OBJECT}:
            log_stage("VISION_REQUEST_START", kind=kind)
            answer = describe_current_object(crop, config, state)
            log_stage("VISION_REQUEST_END", kind=kind)
        else:
            log_stage("VISION_REQUEST_START", kind=kind)
            answer = answer_for(kind, crop, intent, config, state)
            log_stage("VISION_REQUEST_END", kind=kind)
        state.last_core_answer = answer
        state.last_final_answer = answer
        state.last_answer_time = time.monotonic()
        visual_speak_final(answer, state, config, "yes")
        state.last_assistant_question_type = "none"
        state.last_follow_up_offer = None
        maybe_store_scene_memory(kind, crop, answer, config, state, crop_path)
        if kind == "sign":
            record_sign_memory(crop, crop_path, config, state, now=state.last_answer_time)
    finally:
        if state.visual_prompt_state not in {VISUAL_COOLDOWN, VISUAL_RESULT_PLAYING}:
            visual_interaction_complete(state, config, "timeout")
        state.is_busy = False


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 gaze-triggered plant and sign assistant.")
    parser.add_argument("--camera", type=int, help="Legacy local camera index. Prefer --camera-source local --camera-index N.")
    parser.add_argument("--camera-index", type=int, help="Local camera index for --camera-source local.")
    parser.add_argument("--camera-source", default="mentra", help="mentra, local, none, or a stream URL.")
    parser.add_argument("--mic", type=int, help="Use this microphone device index, for example --mic 0.")
    parser.add_argument("--camera-only", action="store_true", help="Open the camera window only, with no wake listener, speech, or AI flow.")
    parser.add_argument("--preview-only", action="store_true", help="Preview the selected camera source only.")
    parser.add_argument("--test-wake", action="store_true", help="Listen for wake phrases only and print wake debug output.")
    parser.add_argument("--network-check", action="store_true", help="Print hotspot, Android, MediaMTX, and stream diagnostics, then exit.")
    return parser.parse_args()


def normalize_camera_source(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def resolve_camera_source(args):
    if args.camera is not None:
        index = args.camera
        app_log.debug("CAMERA_SOURCE\nmode=local\nindex=" + str(index))
        return index, "local"

    source = (args.camera_source or "mentra").strip()
    if source == "mentra":
        app_log.debug("CAMERA_SOURCE\nmode=mentra\nurl=" + DEFAULT_MENTRA_RTSP_URL)
        return DEFAULT_MENTRA_RTSP_URL, "mentra"
    if source == "local":
        if args.camera_index is None:
            raise SystemExit("Use --camera-source local --camera-index N to open a Mac camera.")
        app_log.debug("CAMERA_SOURCE\nmode=local\nindex=" + str(args.camera_index))
        return args.camera_index, "local"
    if source == "none":
        app_log.debug("CAMERA_SOURCE\nmode=none")
        return None, "none"

    normalized = normalize_camera_source(source)
    mode = "local" if isinstance(normalized, int) else "stream"
    detail = f"index={normalized}" if isinstance(normalized, int) else f"url={normalized}"
    app_log.debug(f"CAMERA_SOURCE\nmode={mode}\n{detail}")
    return normalized, mode


def run_wake_test(config, mic_index=None):
    test_config = replace(
        config,
        voice_activation_mode=True,
        wake_mode=True,
        wake_debug_transcripts=True,
        wake_log_empty_transcripts=True,
    )
    print("wake test mode")
    while True:
        transcript = listen_for_wake_phrase(
            mic_index=mic_index if mic_index is not None else test_config.mic_device_index,
            config=test_config,
        )
        if not transcript:
            continue
        if isinstance(transcript, dict):
            transcript = transcript.get("transcript", "")
        cleaned = strip_wake_phrase(
            transcript,
            getattr(test_config, "wake_phrases", ()),
            getattr(test_config, "allow_single_word_wake", False),
        )
        print(f'cleaned command: "{cleaned}"')
        if not cleaned:
            print('wake only: would ask "Yes?"')
        elif is_incomplete_command(cleaned):
            print('wake incomplete: would ask "I did not catch the full question. Please say it again."')


def _overlay_ready_banner(frame, text):
    cv2.putText(frame, text, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)


def main():
    new_interaction("startup")
    log_stage("APP_START")
    args = parse_args()
    config = Config.from_env(camera_index=args.camera, mic_device_index=args.mic)
    import os

    os.environ.setdefault("LATENCY_WARN_THRESHOLD_MS", str(getattr(config, "latency_warn_threshold_ms", 1000)))
    os.environ.setdefault("LATENCY_LOG_FILE_ENABLED", "1" if getattr(config, "latency_log_file_enabled", False) else "0")
    os.environ.setdefault("LOG_LEVEL", getattr(config, "log_level", "INFO"))
    os.environ.setdefault("LATENCY_CONSOLE_MODE", getattr(config, "latency_console_mode", "summary"))
    os.environ.setdefault("TERMINAL_COMPACT_MODE", "1" if getattr(config, "terminal_compact_mode", False) else "0")
    if args.network_check:
        print_network_check(config.android_trail_base_url)
        return
    if args.test_wake:
        run_wake_test(config, mic_index=args.mic)
        return
    camera_source, _camera_mode = resolve_camera_source(args)
    if camera_source is None:
        return
    stream_wait_source = camera_source
    if isinstance(camera_source, str):
        stream_wait_source, camera_source = split_stream_urls(camera_source)

    camera_only_mode = bool(args.camera_only or args.preview_only)
    camera_source_is_stream = isinstance(camera_source, str) and camera_source.lower().startswith(("http://", "https://", "rtmp://", "rtsp://", "udp://", "srt://"))
    wake_enabled = not camera_only_mode and (getattr(config, "voice_activation_mode", True) or getattr(config, "wake_mode", False))

    preload_thread = threading.Thread(target=preload_whisper_model, args=(config,), daemon=True)
    preload_thread.start()
    if wake_enabled:
        preload_thread.join(timeout=20)
    update_runtime_status(whisper=bool(getattr(config, "use_whisper_stt", True)))

    trigger = DwellTrigger(config.dwell_seconds, config.cooldown_seconds)
    state = SessionState(follow_up_timeout_seconds=config.follow_up_timeout_seconds)
    tts_server = TtsHttpServer(host="0.0.0.0", port=8765).start()
    app_log.debug("TTS server listening on http://0.0.0.0:8765/command (for glasses audio)")
    mic_server = None
    rtmp_audio = None
    if getattr(config, "use_glasses_mic", False):
        if camera_source_is_stream and isinstance(camera_source, str) and camera_source.lower().startswith(("rtsp://", "rtmp://")):
            rtmp_audio = RtmpAudioIngest(camera_source).start()
            if rtmp_audio.started:
                print("Glasses mic: pulling audio from RTSP stream")
                app_log.debug(f"MIC_SOURCE\nmode=rtsp\nurl={rtmp_audio.audio_url}")
                log_stage("RTSP_AUDIO_CONNECTED", url=rtmp_audio.audio_url)
                update_runtime_status(audio=True)
        if rtmp_audio is None or not rtmp_audio.started:
            mic_server = MicIngestServer(host="0.0.0.0", port=8767).start()
            print("Glasses mic ingest on http://0.0.0.0:8767/mic/pcm (BLE PCM fallback)")
            app_log.debug("MIC_SOURCE\nmode=http_pcm\nendpoint=0.0.0.0:8767/mic/pcm")
            update_runtime_status(audio=True)
    elif wake_enabled:
        print("Using Mac microphone — set USE_GLASSES_MIC=1 in .env for glasses mic")
    if wake_enabled and getattr(config, "use_glasses_mic", False):
        calibrate_wake_ambient(config)
    phone_ok = False
    phone_ok, _ = android_bridge_diagnostics(config)
    update_runtime_status(androidBridge=phone_ok)
    if phone_ok:
        log_stage("ANDROID_BRIDGE_CONNECTED")
    phone_health_next_check = time.monotonic() + 15.0
    wake_queue = queue.SimpleQueue()
    stop_event = threading.Event()
    last_analysis_signature = None
    last_crop_log_at = 0.0
    last_crop_log_kind = ""
    wake_thread = None
    if wake_enabled:
        wake_thread = threading.Thread(target=wake_listener_loop, args=(config, state, wake_queue, stop_event), daemon=True)
        wake_thread.name = "WakeListener"
        wake_thread.start()

    try:
        if camera_source_is_stream and not wait_for_publisher(
            stream_wait_source,
            ingest_source=camera_source if camera_source != stream_wait_source else None,
        ):
            return
        if camera_source_is_stream:
            log_stage("VIDEO_STREAM_READY")

        with Camera(
            camera_source,
            fallback_source=stream_wait_source if stream_wait_source != camera_source else None,
        ) as camera:
            if not camera.opened:
                print(f"Could not open camera source {camera_source!r}. Try python main.py to scan cameras, or use python main.py --camera 1.")
                return

            ready_announced = False
            stream_misses = 0
            max_stream_misses = 600
            while True:
                if config.android_trail_base_url and time.monotonic() >= phone_health_next_check:
                    phone_health_next_check = time.monotonic() + 15.0
                    now_ok, _ = check_trail_health(config)
                    if now_ok and not phone_ok:
                        print(f"Phone trail server now reachable at {config.android_trail_base_url}")
                    phone_ok = now_ok
                    update_runtime_status(androidBridge=phone_ok)
                if not camera_only_mode and not state.is_busy:
                    while True:
                        try:
                            wake_result = wake_queue.get_nowait()
                        except queue.Empty:
                            break
                        if isinstance(wake_result, dict):
                            transcript = wake_result.get("transcript", "")
                            set_interaction(
                                wake_result.get("interactionId") or new_interaction(),
                                started_at=wake_result.get("startedAt"),
                                stages=wake_result.get("stages"),
                            )
                        else:
                            transcript = wake_result
                            new_interaction()
                        handle_voice_command(transcript, config, state, camera)

                frame = camera.read()
                if frame is None:
                    if camera_source_is_stream:
                        stream_misses += 1
                        if stream_misses >= max_stream_misses:
                            print(
                                f"Stream frame was unavailable from {camera_source!r} for too long. "
                                "Phone may have stopped publishing — reopen Glasses -> Start everything."
                            )
                            return
                        if stream_misses % 100 == 1:
                            print("Stream gap — waiting for frames (keep phone Mentra session running)...")
                        time.sleep(0.02)
                        continue
                    print("Camera frame was unavailable. Check macOS camera permission and try again.")
                    return
                stream_misses = 0
                update_runtime_status(mentraStream=camera_source_is_stream, video=True)

                if camera_only_mode:
                    if not ready_announced:
                        print("READY: camera-only preview is live. Press q to quit.")
                        log_stage("SYSTEM_READY")
                        ready_announced = True
                    _overlay_ready_banner(frame, "READY - camera-only demo")
                    if camera.show(frame):
                        break
                    continue

                crop, box = focus_crop(frame, config.focus_fraction)
                signature = crop_signature(crop)
                if not ready_announced:
                    ready_parts = ["camera preview is live"]
                    if wake_enabled:
                        ready_parts.append("wake listener active")
                    ready_parts.append("analysis running")
                    mic_mode = "rtsp" if rtmp_audio and rtmp_audio.started else "http_pcm" if mic_server else "mac"
                    print(f"SYSTEM ready camera={_camera_mode} mic={mic_mode} android={'connected' if phone_ok else 'unavailable'}")
                    log_stage("SYSTEM_READY")
                    ready_announced = True
                if not state.is_busy and trigger.ready_to_analyze(config.analysis_interval):
                    if crop_changed_enough(signature, last_analysis_signature):
                        last_analysis_signature = signature
                        result = analyze_crop(crop, config)
                        now = time.monotonic()
                        if result.kind not in {"plant", "sign"}:
                            visual_mark_absence(result.kind, config, state, crop=crop, now=now)
                        if result.kind in {"plant", "sign"} or result.kind != last_crop_log_kind or now - last_crop_log_at >= getattr(config, "vision_log_interval_seconds", 10):
                            app_log.debug(f"center crop: {result.kind} ({result.reason})")
                            last_crop_log_at = now
                            last_crop_log_kind = result.kind
                        if trigger.update(result.kind):
                            if not visual_prompt_should_start(result.kind, crop, config, state, now=now):
                                continue
                            crop_path = save_debug_crop(result.kind, crop)
                            remembered = None
                            if getattr(config, "scene_memory_enabled", True):
                                remembered = lookup_scene_memory(
                                    result.kind,
                                    "scene",
                                    crop,
                                    ttl_seconds=getattr(config, "scene_memory_ttl_seconds", 3600),
                                )
                            if remembered:
                                print("scene memory: using stored answer")
                                seen_count = int(remembered.get("seen_count", 2))
                                state.last_detected_kind = result.kind
                                state.last_crop_path = crop_path
                                answer = format_remembered_answer(
                                    remembered.get("answer") or "",
                                    result.kind,
                                    seen_count,
                                )
                                state.last_core_answer = answer
                                state.last_final_answer = answer
                                state.last_answer_time = time.monotonic()
                                state.last_vision_result = remembered.get("vision_result")
                                state.last_plant_id_result = restore_plant_id(remembered.get("plant_id_result"))
                                visual_speak_final(answer, state, config, "yes")
                                if result.kind == "sign" or state.last_detected_kind == "sign":
                                    record_sign_memory(crop, crop_path, config, state, now=state.last_answer_time)
                                draw_focus_box(frame, box)
                                if camera.show(frame):
                                    break
                                continue
                            if result.kind == "sign":
                                suppressed, cached_answer, _, memory_match = maybe_suppress_sign_prompt(crop, crop_path, config, state)
                                if suppressed:
                                    if cached_answer:
                                        seen_count = int((memory_match or {}).get("seen_count", 2))
                                        state.last_detected_kind = "sign"
                                        state.last_crop_path = crop_path
                                        answer = format_remembered_answer(cached_answer, "sign", seen_count)
                                        state.last_core_answer = answer
                                        state.last_final_answer = answer
                                        state.last_answer_time = time.monotonic()
                                        visual_speak_final(answer, state, config, "yes")
                                    else:
                                        visual_interaction_complete(state, config, "no")
                                    draw_focus_box(frame, box)
                                    if camera.show(frame):
                                        break
                                    continue
                            state.is_busy = True
                            threading.Thread(
                                target=handle_trigger,
                                args=(result.kind, crop.copy(), config, state, camera),
                                kwargs={"crop_path": crop_path},
                                daemon=True,
                            ).start()
                draw_focus_box(frame, box)
                if camera.show(frame):
                    break
    finally:
        stop_event.set()
        if rtmp_audio is not None:
            rtmp_audio.stop()
        if mic_server is not None:
            mic_server.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
