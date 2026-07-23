from dataclasses import dataclass


@dataclass
class SessionState:
    last_detected_kind: str | None = None
    last_crop_path: str | None = None
    last_prompt: str | None = None
    last_user_transcript: str | None = None
    last_action: str | None = None
    last_vision_result: dict | None = None
    last_plant_id_result: object | None = None
    last_final_answer: str | None = None
    last_core_answer: str | None = None
    last_follow_up_offer: str | None = None
    last_response_cache_key: str | None = None
    last_response_cache_value: str | None = None
    last_response_cache_time: float = 0.0
    last_answer_time: float = 0.0
    last_assistant_question_type: str = "none"
    follow_up_enabled: bool = False
    awaiting_follow_up_reply: bool = False
    follow_up_timeout_seconds: float = 1.5
    is_busy: bool = False
    is_processing_command: bool = False
    is_speaking: bool = False
    visual_prompt_state: str = "IDLE"
    visual_interaction_id: str = ""
    visual_interaction_count: int = 0
    visual_object_kind: str | None = None
    visual_object_signature: object | None = None
    visual_object_present: bool = False
    visual_object_last_seen: float = 0.0
    visual_object_absent_since: float = 0.0
    visual_cooldown_until: float = 0.0
    visual_reply_retries: int = 0
