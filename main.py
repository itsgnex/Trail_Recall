import argparse
from datetime import datetime
from dataclasses import replace
from pathlib import Path
import queue
import threading
import time
#Build a confidence-aware, landmark-augmented route retracing assistant with a simple junction decision feature.
import cv2
import numpy as np

from src.camera import Camera, choose_camera
from src.config import Config
from src.intent import Intent, classify_intent_with_source
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
)
from src.session_state import SessionState
from src.scene_memory import lookup_scene_memory, restore_plant_id
from src.sign_memory import add_or_update_sign_memory, cleanup_old_signs, find_similar_sign
from src.speech_in import is_incomplete_command, is_wake_check_command, listen, listen_for_wake_phrase, strip_wake_phrase
from src.speech_out import is_speaking, speak
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


def crop_signature(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return small


def crop_changed_enough(current, previous, threshold=5.0):
    if previous is None:
        return True
    return float(np.mean(np.abs(current.astype(np.int16) - previous.astype(np.int16)))) >= threshold


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
        return False, None, None
    embedding = get_crop_embedding(crop)
    if embedding is None:
        return False, None, None

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
        return False, None, embedding

    print(f"sign memory: possible duplicate similarity={similarity:.2f}")
    if similarity < getattr(config, "sign_clip_duplicate_threshold", 0.88):
        return False, None, embedding

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
            return True, match.get("final_answer") or None, embedding
        print("sign memory: different sign, continuing")
        return False, None, embedding

    print("sign memory: duplicate threshold reached, continuing without verify")
    return False, None, embedding


def listen_for_follow_up_reply(prompt, config, state, detected_kind, ocr_text="", record_seconds=None):
    state.last_assistant_question_type = "follow_up_offer"
    state.follow_up_enabled = True
    state.awaiting_follow_up_reply = True
    try:
        heard = listen(
            config,
            typed_fallback=False,
            record_seconds=record_seconds or getattr(config, "follow_up_timeout_seconds", 8),
            label="recording follow-up",
        )
        if not heard:
            return "", None, None

        intent, source = classify_intent_with_source(
            heard,
            config,
            detected_kind,
            prompt,
            ocr_text,
            last_question_type=state.last_assistant_question_type,
        )
        print(f'follow-up said: "{heard}" -> {intent.value} via {source}')
        state.last_user_transcript = heard
        state.last_action = intent.value
        return heard, intent, source
    finally:
        state.follow_up_enabled = False
        state.awaiting_follow_up_reply = False


def handle_voice_command(transcript, config, state, camera):
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
                record_seconds=getattr(config, "command_record_seconds", 6),
                label="recording command",
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
                record_seconds=getattr(config, "command_record_seconds", 6),
                label="recording command",
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
        print(f'command transcript: "{command}"')
        if not command or is_incomplete_command(command):
            speak(get_wake_check_response())
            return

        intent, source = classify_intent_with_source(
            command,
            config,
            state.last_detected_kind,
            state.last_prompt or "",
            last_question_type=state.last_assistant_question_type,
        )
        print(f'voice said: "{command}" -> {intent.value} via {source}')
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
            if crop_kind == "sign":
                state.last_detected_kind = "sign"
                answer = answer_for("sign", crop, intent, config, state)
            elif crop_kind == "plant":
                state.last_detected_kind = "plant"
                answer = answer_for("plant", crop, intent, config, state)
            else:
                answer = describe_current_object(crop, config, state)

            state.last_crop_path = crop_path
            state.last_core_answer = answer
            state.last_final_answer = answer
            state.last_answer_time = time.monotonic()
            speak(answer)
            if state.last_detected_kind == "sign" or crop_kind == "sign":
                record_sign_memory(crop, crop_path, config, state, now=state.last_answer_time)
            return

        clarification_prompt = get_clarification_response()
        state.awaiting_follow_up_reply = True
        speak(clarification_prompt)
        heard, follow_up_intent, follow_up_source = listen_for_follow_up_reply(
            clarification_prompt,
            config,
            state,
            state.last_detected_kind,
            state.last_prompt or "",
            record_seconds=getattr(config, "follow_up_timeout_seconds", 8),
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
            if crop_kind == "sign":
                state.last_detected_kind = "sign"
                answer = answer_for("sign", crop, intent, config, state)
            elif crop_kind == "plant":
                state.last_detected_kind = "plant"
                answer = answer_for("plant", crop, intent, config, state)
            else:
                answer = describe_current_object(crop, config, state)
            state.last_crop_path = crop_path
            state.last_core_answer = answer
            state.last_final_answer = answer
            state.last_answer_time = time.monotonic()
            speak(answer)
            if state.last_detected_kind == "sign" or crop_kind == "sign":
                record_sign_memory(crop, crop_path, config, state, now=state.last_answer_time)
            return
        speak(get_clarification_response())
        return
    finally:
        state.is_processing_command = False
        state.is_busy = False


def wake_listener_loop(config, state, wake_queue, stop_event):
    if not (getattr(config, "voice_activation_mode", True) or getattr(config, "wake_mode", False)):
        return
    while not stop_event.is_set():
        if getattr(config, "pause_wake_during_tts", True) and (is_speaking() or getattr(state, "is_busy", False) or getattr(state, "is_processing_command", False) or getattr(state, "follow_up_enabled", False) or getattr(state, "awaiting_follow_up_reply", False)):
            time.sleep(0.1)
            continue
        transcript = listen_for_wake_phrase(mic_index=getattr(config, "mic_device_index", 0), config=config, state=state)
        if transcript:
            if getattr(state, "is_busy", False):
                time.sleep(0.1)
                continue
            print("voice activation: wake phrase detected")
            wake_queue.put(transcript)
            time.sleep(max(0.1, float(getattr(config, "wake_cooldown_seconds", 2))))
        else:
            time.sleep(0.05)


def handle_follow_up(crop, config, state, camera):
    if not getattr(config, "follow_up_mode", True):
        return
    if state.last_assistant_question_type != "follow_up_offer":
        return

    state.follow_up_enabled = True
    state.follow_up_timeout_seconds = getattr(config, "follow_up_timeout_seconds", 8)
    try:
        for _ in range(getattr(config, "max_follow_up_turns", 2)):
            heard = listen(config, typed_fallback=False, record_seconds=state.follow_up_timeout_seconds)
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
                clarification_prompt = get_clarification_response()
                state.awaiting_follow_up_reply = True
                speak(clarification_prompt)
                heard, follow_up_intent, follow_up_source = listen_for_follow_up_reply(
                    clarification_prompt,
                    config,
                    state,
                    state.last_detected_kind,
                    state.last_final_answer or "",
                    record_seconds=getattr(config, "follow_up_timeout_seconds", 8),
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
        speak(prompt)
        state.awaiting_follow_up_reply = True
        try:
            heard = listen(config)
        finally:
            state.awaiting_follow_up_reply = False
        ocr_text = read_text(crop) if kind == "sign" else ""
        intent, source = classify_intent_with_source(heard, config, kind, prompt, ocr_text, last_question_type=state.last_assistant_question_type)
        print(f'user said: "{heard or "[nothing heard]"}" -> {intent.value} via {source}')
        state.last_user_transcript = heard
        state.last_action = intent.value

        if intent == Intent.REPEAT_LAST_MESSAGE:
            speak(get_repeat_response())
            speak(prompt)
            heard = listen(config)
            intent, source = classify_intent_with_source(heard, config, kind, prompt, ocr_text, last_question_type=state.last_assistant_question_type)
            print(f'user said: "{heard or "[nothing heard]"}" -> {intent.value} via {source}')
            state.last_user_transcript = heard
            state.last_action = intent.value

        if intent in {Intent.CANCEL, Intent.STOP_LISTENING}:
            speak(get_cancel_response())
            state.last_assistant_question_type = "none"
            return
        if intent == Intent.ASK_CLARIFICATION:
            clarification_prompt = get_clarification_response()
            state.awaiting_follow_up_reply = True
            speak(clarification_prompt)
            heard, follow_up_intent, follow_up_source = listen_for_follow_up_reply(
                clarification_prompt,
                config,
                state,
                kind,
                ocr_text,
                record_seconds=getattr(config, "follow_up_timeout_seconds", 8),
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
        if intent == Intent.SPEAK_SLOWER:
            speak("Of course. I’ll keep it slower and brief.")

        if intent in {Intent.WHAT_AM_I_LOOKING_AT, Intent.DESCRIBE_CURRENT_OBJECT, Intent.IDENTIFY_CURRENT_OBJECT}:
            answer = describe_current_object(crop, config, state)
        else:
            answer = answer_for(kind, crop, intent, config, state)
        state.last_core_answer = answer
        state.last_final_answer = answer
        state.last_answer_time = time.monotonic()
        speak(answer)
        if needs_clearer_view(answer):
            state.last_assistant_question_type = "retry_clearer_view"
            return
        state.last_follow_up_offer = None
        if answer_asks_follow_up(answer):
            state.last_assistant_question_type = "follow_up_offer"
        elif getattr(config, "speak_follow_up_offer", False) and intent in {Intent.EXPLAIN_PLANT, Intent.EXPLAIN_SIGN_MEANING, Intent.WHAT_AM_I_LOOKING_AT, Intent.DESCRIBE_CURRENT_OBJECT, Intent.IDENTIFY_CURRENT_OBJECT}:
            state.last_follow_up_offer = get_follow_up_offer()
            speak(state.last_follow_up_offer)
            state.last_assistant_question_type = "follow_up_offer"
        else:
            state.last_assistant_question_type = "none"
        handle_follow_up(crop, config, state, camera)
        if kind == "sign":
            record_sign_memory(crop, crop_path, config, state, now=state.last_answer_time)
    finally:
        state.is_busy = False


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 gaze-triggered plant and sign assistant.")
    parser.add_argument("--camera", type=int, help="Open this camera index directly, for example --camera 1.")
    parser.add_argument("--camera-source", help="Open a camera index or stream URL directly, for example --camera-source rtmp://127.0.0.1:1935/live/mentra-live or --camera-source http://127.0.0.1:8888/live/mentra-live/index.m3u8.")
    parser.add_argument("--mic", type=int, help="Use this microphone device index, for example --mic 0.")
    parser.add_argument("--camera-only", action="store_true", help="Open the camera window only, with no wake listener, speech, or AI flow.")
    parser.add_argument("--test-wake", action="store_true", help="Listen for wake phrases only and print wake debug output.")
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
    args = parse_args()
    config = Config.from_env(camera_index=args.camera, mic_device_index=args.mic)
    if args.test_wake:
        run_wake_test(config, mic_index=args.mic)
        return
    camera_source = normalize_camera_source(
        args.camera_source if args.camera_source is not None else (config.camera_index if args.camera is not None else choose_camera())
    )
    if camera_source is None:
        return

    camera_only_mode = bool(args.camera_only)
    camera_source_is_stream = isinstance(camera_source, str) and camera_source.lower().startswith(("http://", "https://", "rtmp://", "rtsp://", "udp://", "srt://"))
    wake_enabled = not camera_only_mode and (getattr(config, "voice_activation_mode", True) or getattr(config, "wake_mode", False))

    trigger = DwellTrigger(config.dwell_seconds, config.cooldown_seconds)
    state = SessionState(follow_up_timeout_seconds=config.follow_up_timeout_seconds)
    wake_queue = queue.SimpleQueue()
    stop_event = threading.Event()
    last_analysis_signature = None
    wake_thread = None
    if wake_enabled:
        wake_thread = threading.Thread(target=wake_listener_loop, args=(config, state, wake_queue, stop_event), daemon=True)
        wake_thread.start()

    try:
        with Camera(camera_source) as camera:
            if not camera.opened:
                print(f"Could not open camera source {camera_source!r}. Try python main.py to scan cameras, or use python main.py --camera 1.")
                return

            ready_announced = False
            while True:
                if not camera_only_mode and not state.is_busy:
                    while True:
                        try:
                            transcript = wake_queue.get_nowait()
                        except queue.Empty:
                            break
                        handle_voice_command(transcript, config, state, camera)

                frame = camera.read()
                if frame is None:
                    if camera_source_is_stream:
                        print(f"Stream frame was unavailable from {camera_source!r}. Check that the Mentra RTMP/HLS publisher is live and the URL is correct.")
                    else:
                        print("Camera frame was unavailable. Check macOS camera permission and try again.")
                    return

                if camera_only_mode:
                    if not ready_announced:
                        print("READY: camera-only preview is live. Press q to quit.")
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
                    print("READY: " + ", ".join(ready_parts) + ".")
                    ready_announced = True
                if not state.is_busy and trigger.ready_to_analyze(config.analysis_interval):
                    if crop_changed_enough(signature, last_analysis_signature):
                        last_analysis_signature = signature
                        result = analyze_crop(crop, config)
                        print(f"center crop: {result.kind} ({result.reason})")
                        if trigger.update(result.kind):
                            crop_path = save_debug_crop(result.kind, crop)
                            remembered = lookup_scene_memory(
                                result.kind,
                                "scene",
                                crop,
                                ttl_seconds=getattr(config, "scene_memory_ttl_seconds", 3600),
                            )
                            if remembered:
                                print("scene memory: using stored answer")
                                state.last_detected_kind = result.kind
                                state.last_crop_path = crop_path
                                state.last_core_answer = remembered.get("answer") or ""
                                state.last_final_answer = remembered.get("answer") or ""
                                state.last_answer_time = time.monotonic()
                                state.last_vision_result = remembered.get("vision_result")
                                state.last_plant_id_result = restore_plant_id(remembered.get("plant_id_result"))
                                speak(remembered.get("answer") or "")
                                if result.kind == "sign" or state.last_detected_kind == "sign":
                                    record_sign_memory(crop, crop_path, config, state, now=state.last_answer_time)
                                draw_focus_box(frame, box)
                                if camera.show(frame):
                                    break
                                continue
                            if result.kind == "sign":
                                suppressed, cached_answer, _ = maybe_suppress_sign_prompt(crop, crop_path, config, state)
                                if suppressed:
                                    if cached_answer:
                                        state.last_detected_kind = "sign"
                                        state.last_crop_path = crop_path
                                        state.last_core_answer = cached_answer
                                        state.last_final_answer = cached_answer
                                        state.last_answer_time = time.monotonic()
                                        speak(cached_answer)
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
