from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from . import app_log

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "assets" / "common_tts"
MANIFEST_PATH = AUDIO_DIR / "manifest.json"
TEMPLATE_DIR = AUDIO_DIR / "templates"

COMMON_PHRASES = [
    ("general_yes_hear_you", "GENERAL", "Yes, I can hear you."),
    ("general_yes_short", "GENERAL", "Yes?"),
    ("general_can_hear_tell_look", "GENERAL", "I can hear you. Please tell me what you want me to look at."),
    ("general_what_look_at", "GENERAL", "What would you like me to look at?"),
    ("general_say_again", "GENERAL", "Please say that again."),
    ("general_full_question_again", "GENERAL", "I did not catch the full question. Please say it again."),
    ("general_missed_continue", "GENERAL", "I may have missed that. Would you like me to continue?"),
    ("general_unsure_continue_explain", "GENERAL", "I’m not sure if you wanted me to continue. Should I explain it?"),
    ("general_okay", "GENERAL", "Okay."),
    ("general_no_problem", "GENERAL", "No problem."),
    ("general_wait_moment", "GENERAL", "Please wait a moment."),
    ("general_ready", "GENERAL", "I’m ready."),
    ("general_listening", "GENERAL", "I’m listening."),
    ("general_yes_listening_check", "GENERAL", "Yes, I’m listening. What would you like me to check?"),
    ("general_tell_me_check", "GENERAL", "Please tell me what you want me to check."),
    ("general_unsure_help_this", "GENERAL", "I’m sorry, I wasn’t sure what you wanted. Would you like me to help with this?"),
    ("general_didnt_catch_watch_quietly", "GENERAL", "I didn’t catch that. I’ll go back to watching quietly."),
    ("general_center_not_clear_move_closer", "GENERAL", "I can see something in the center, but it is not clear enough yet. Try holding it steadier or moving a little closer."),
    ("sign_may_see_help", "SIGNS", "I may be seeing a sign here. Would you like help with it?"),
    ("sign_should_explain_saying", "SIGNS", "Should I explain what this sign is saying?"),
    ("sign_try_tell_means", "SIGNS", "I can try to tell you what this sign means."),
    ("sign_cannot_read_clearly", "SIGNS", "I’m not able to read this sign clearly from here."),
    ("sign_move_closer", "SIGNS", "Please move a little closer to the sign."),
    ("sign_keep_centered", "SIGNS", "Please keep the sign centered in front of you."),
    ("sign_stop", "SIGNS", "This appears to be a stop sign."),
    ("sign_warning", "SIGNS", "This appears to be a warning sign."),
    ("sign_information", "SIGNS", "This appears to be an information sign."),
    ("sign_not_confident", "SIGNS", "I cannot identify this sign confidently."),
    ("plant_may_see_identify", "PLANTS", "I may be seeing a plant. Would you like me to identify it?"),
    ("plant_move_closer", "PLANTS", "Please move a little closer to the plant."),
    ("plant_keep_centered", "PLANTS", "Please keep the plant centered in front of you."),
    ("plant_checking", "PLANTS", "I’m checking the plant."),
    ("plant_not_confident", "PLANTS", "I could not identify this plant confidently."),
    ("plant_image_not_clear", "PLANTS", "The image is not clear enough to identify the plant."),
    ("plant_more_info", "PLANTS", "Would you like more information about this plant?"),
    ("trail_started", "TRAIL RECORDING", "Got it. Trail recording started."),
    ("trail_already_active", "TRAIL RECORDING", "Trail recording is already active."),
    ("trail_saved_all_set", "TRAIL RECORDING", "Trail saved. You’re all set."),
    ("trail_none_to_stop", "TRAIL RECORDING", "There is no active trail to stop."),
    ("trail_recording", "TRAIL RECORDING", "Your route is being recorded."),
    ("trail_saved_success", "TRAIL RECORDING", "The trail has been saved successfully."),
    ("return_starting_back", "RETURN NAVIGATION", "Starting navigation back."),
    ("return_turn_around_continue", "RETURN NAVIGATION", "Turn around and continue straight."),
    ("return_continue_straight", "RETURN NAVIGATION", "Please continue straight."),
    ("return_keep_straight", "RETURN NAVIGATION", "Keep going straight."),
    ("return_turn_left", "RETURN NAVIGATION", "Turn left."),
    ("return_turn_right", "RETURN NAVIGATION", "Turn right."),
    ("return_stay_left", "RETURN NAVIGATION", "Stay left."),
    ("return_stay_right", "RETURN NAVIGATION", "Stay right."),
    ("return_decision_point", "RETURN NAVIGATION", "You are approaching a decision point."),
    ("return_choose_left_right", "RETURN NAVIGATION", "Please choose left or right."),
    ("return_wrong_way", "RETURN NAVIGATION", "You are going the wrong way."),
    ("return_turn_around", "RETURN NAVIGATION", "Please turn around."),
    ("return_back_on_route", "RETURN NAVIGATION", "You are back on the route."),
    ("return_destination_reached", "RETURN NAVIGATION", "You have reached the destination."),
    ("return_starting_point", "RETURN NAVIGATION", "You have returned to the starting point."),
    ("return_navigation_complete", "RETURN NAVIGATION", "Navigation is complete."),
    ("return_location_not_accurate", "RETURN NAVIGATION", "Your location is not accurate enough yet."),
    ("return_finding_location", "RETURN NAVIGATION", "Please wait while I find your location."),
    ("return_no_safe_route", "RETURN NAVIGATION", "I cannot calculate a safe return route right now."),
    ("confirm_continue", "CONFIRMATIONS", "Would you like me to continue?"),
    ("confirm_explain", "CONFIRMATIONS", "Would you like me to explain it?"),
    ("confirm_yes", "CONFIRMATIONS", "Did you say yes?"),
    ("confirm_no", "CONFIRMATIONS", "Did you say no?"),
    ("confirm_yes_or_no", "CONFIRMATIONS", "Please answer yes or no."),
]

TEMPLATES = [
    ("template_turn_left_distance", "RETURN NAVIGATION", "Turn left in {distance} metres.", r"^turn left in (?P<distance>[\w. -]+) metres$"),
    ("template_turn_right_distance", "RETURN NAVIGATION", "Turn right in {distance} metres.", r"^turn right in (?P<distance>[\w. -]+) metres$"),
    ("template_continue_distance", "RETURN NAVIGATION", "Continue straight for {distance} metres.", r"^continue straight for (?P<distance>[\w. -]+) metres$"),
    ("template_alternate_minutes", "RETURN NAVIGATION", "The alternate route takes about {minutes} minutes.", r"^the alternate route takes about (?P<minutes>[\w. -]+) minutes$"),
    ("template_route_adds_minutes", "RETURN NAVIGATION", "This route adds about {minutes} minutes.", r"^this route adds about (?P<minutes>[\w. -]+) minutes$"),
]


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").lower().strip()
    value = value.replace("’", "'").replace("‘", "'").replace("`", "'")
    value = re.sub(r"[^\w\s']", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def wav_filename(phrase_id: str, text: str | None = None) -> str:
    suffix = ""
    if text is not None:
        suffix = "-" + hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:10]
    return f"{phrase_id}{suffix}.wav"


def base_manifest() -> dict:
    return {
        "version": 1,
        "sampleRate": 16000,
        "channels": 1,
        "phrases": [
            {
                "phraseId": phrase_id,
                "category": category,
                "text": text,
                "normalizedText": normalize_text(text),
                "wav": wav_filename(phrase_id),
                "wavFilename": wav_filename(phrase_id),
            }
            for phrase_id, category, text in COMMON_PHRASES
        ],
        "templates": [
            {
                "phraseId": phrase_id,
                "category": category,
                "text": template,
                "normalizedText": normalize_text(template),
                "wav": "",
                "wavFilename": "",
                "pattern": pattern,
            }
            for phrase_id, category, template, pattern in TEMPLATES
        ],
        "generatedTemplates": [],
    }


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return base_manifest()
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_manifest(manifest: dict | None = None) -> dict:
    manifest = manifest or load_manifest()
    errors = []
    seen = set()
    for entry in manifest.get("phrases", []):
        for key in ("phraseId", "category", "text", "normalizedText"):
            if not entry.get(key):
                errors.append(f"missing {key}: {entry}")
        if not (entry.get("wav") or entry.get("wavFilename")):
            errors.append(f"missing wav filename: {entry}")
        if entry.get("phraseId") in seen:
            errors.append(f"duplicate phraseId: {entry.get('phraseId')}")
        seen.add(entry.get("phraseId"))
        if normalize_text(entry.get("text", "")) != entry.get("normalizedText"):
            errors.append(f"bad normalizedText: {entry.get('phraseId')}")
    return {"ok": not errors, "errors": errors, "phrases": len(manifest.get("phrases", []))}


def _log_common(kind: str, phrase_id: str = "", wav_path: Path | str = "", started: float = 0.0):
    latency = int((time.monotonic() - started) * 1000) if started else 0
    extra = " ttsGenerationMs=0" if kind.endswith("_HIT") else ""
    if kind.endswith("_HIT"):
        app_log.info(f"{kind} phraseId={phrase_id} lookupMs={latency}{extra}")
    else:
        app_log.debug(f"{kind} phraseId={phrase_id} lookupMs={latency}")


def lookup_by_phrase_id(phrase_id: str) -> dict | None:
    started = time.monotonic()
    manifest = load_manifest()
    for section in ("phrases", "generatedTemplates"):
        for entry in manifest.get(section, []):
            if entry.get("phraseId") == phrase_id:
                path = AUDIO_DIR / (entry.get("wav") or entry.get("wavFilename"))
                if path.exists():
                    _log_common("COMMON_AUDIO_HIT", phrase_id, path, started)
                    return {**entry, "path": path}
    _log_common("COMMON_AUDIO_MISS", phrase_id, "", started)
    return None


def lookup_by_text(text: str) -> dict | None:
    started = time.monotonic()
    normalized = normalize_text(text)
    manifest = load_manifest()
    for entry in manifest.get("phrases", []):
        if entry.get("normalizedText") == normalized:
            path = AUDIO_DIR / (entry.get("wav") or entry.get("wavFilename"))
            if path.exists():
                _log_common("COMMON_AUDIO_HIT", entry["phraseId"], path, started)
                return {**entry, "path": path}
    for entry in manifest.get("generatedTemplates", []):
        if entry.get("normalizedText") == normalized:
            path = AUDIO_DIR / (entry.get("wav") or entry.get("wavFilename"))
            if path.exists():
                _log_common("COMMON_AUDIO_TEMPLATE_HIT", entry["phraseId"], path, started)
                return {**entry, "path": path}
    _log_common("COMMON_AUDIO_MISS", "", "", started)
    return None


def template_match(text: str) -> dict | None:
    normalized = normalize_text(text)
    for phrase_id, category, template, pattern in TEMPLATES:
        if re.match(pattern, normalized):
            final_id = wav_filename(phrase_id, normalized)[:-4]
            return {
                "phraseId": final_id,
                "templateId": phrase_id,
                "category": category,
                "text": text.strip(),
                "normalizedText": normalized,
                "wav": f"templates/{wav_filename(phrase_id, normalized)}",
                "wavFilename": f"templates/{wav_filename(phrase_id, normalized)}",
                "template": template,
            }
    return None


def upsert_generated_template(entry: dict) -> None:
    manifest = load_manifest()
    generated = [item for item in manifest.get("generatedTemplates", []) if item.get("normalizedText") != entry["normalizedText"]]
    generated.append({key: entry[key] for key in ("phraseId", "templateId", "category", "text", "normalizedText", "wav", "wavFilename", "template")})
    manifest["generatedTemplates"] = generated
    save_manifest(manifest)
