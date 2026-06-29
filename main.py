import argparse
from datetime import datetime
from pathlib import Path

import cv2

from src.camera import Camera, choose_camera
from src.config import Config
from src.intent import Intent, classify_intent_with_source
from src.llm import answer_for
from src.ocr import read_text
from src.phrases import (
    get_cancel_response,
    get_clarification_response,
    get_prompt,
    get_repeat_response,
    get_unclear_plant_response,
    get_unclear_sign_response,
)
from src.speech_in import listen
from src.speech_out import speak
from src.trigger import DwellTrigger
from src.vision import analyze_crop, draw_focus_box, focus_crop


def save_debug_crop(kind, crop):
    path = Path("debug_crops") / f"{kind}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.jpg"
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), crop)
    print(f"debug crop saved: {path}")


def handle_trigger(kind, crop, config):
    save_debug_crop(kind, crop)
    if kind == "plant":
        prompt = get_prompt("plant")
    elif kind == "sign":
        prompt = get_prompt("sign")
    else:
        prompt = get_prompt("unclear")

    speak(prompt)
    heard = listen(config)
    ocr_text = read_text(crop) if kind == "sign" else ""
    intent, source = classify_intent_with_source(heard, config, kind, prompt, ocr_text)
    print(f'user said: "{heard or "[nothing heard]"}" -> {intent.value} via {source}')

    if intent == Intent.REPEAT_LAST_MESSAGE:
        speak(get_repeat_response())
        speak(prompt)
        heard = listen(config)
        intent, source = classify_intent_with_source(heard, config, kind, prompt, ocr_text)
        print(f'user said: "{heard or "[nothing heard]"}" -> {intent.value} via {source}')

    if intent == Intent.CANCEL:
        speak(get_cancel_response())
        return
    if intent == Intent.ASK_CLARIFICATION:
        speak(get_clarification_response())
        return
    if intent == Intent.SPEAK_SLOWER:
        speak("Of course. I’ll keep it slower and brief.")

    answer = answer_for(kind, crop, intent, config)
    speak(answer)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 gaze-triggered plant and sign assistant.")
    parser.add_argument("--camera", type=int, help="Open this camera index directly, for example --camera 1.")
    parser.add_argument("--mic", type=int, help="Use this microphone device index, for example --mic 0.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config.from_env(camera_index=args.camera, mic_device_index=args.mic)
    camera_index = config.camera_index if args.camera is not None else choose_camera()
    if camera_index is None:
        return

    trigger = DwellTrigger(config.dwell_seconds, config.cooldown_seconds)

    with Camera(camera_index) as camera:
        if not camera.opened:
            print(f"Could not open webcam index {camera_index}. Try python main.py to scan cameras, or use python main.py --camera 1.")
            return

        while True:
            frame = camera.read()
            if frame is None:
                print("Camera frame was unavailable. Check macOS camera permission and try again.")
                return

            crop, box = focus_crop(frame, config.focus_fraction)
            if trigger.ready_to_analyze(config.analysis_interval):
                result = analyze_crop(crop, config)
                print(f"center crop: {result.kind} ({result.reason})")
                if trigger.update(result.kind):
                    handle_trigger(result.kind, crop, config)
            draw_focus_box(frame, box)
            if camera.show(frame):
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
