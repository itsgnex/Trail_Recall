from dataclasses import dataclass
import os


def env_bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    focus_fraction: float = 0.45
    analysis_interval: float = 0.75
    dwell_seconds: float = 2.0
    cooldown_seconds: float = 9.0
    use_clip: bool = True
    use_llm_intent: bool = True
    use_ollama: bool = True
    dialogue_model: str = "gemma3:1b"
    vision_llm_model: str = "gemma3:4b"
    final_response_model: str = "gemma3:4b"
    main_llm_model: str = "gemma3:1b"
    ollama_model: str = "gemma3:4b"
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_dialogue_timeout: int = 30
    ollama_text_timeout: int = 60
    ollama_image_timeout: int = 180
    plant_id_provider: str = "plantnet"
    plantnet_api_key: str = ""
    plantnet_project: str = "all"
    plantnet_lang: str = "en"
    plantnet_organ: str = "auto"
    plantnet_min_confidence: float = 0.45
    plantnet_high_confidence: float = 0.70
    use_plant_id: bool = False
    mic_device_index: int = 0
    mic_listen_timeout: int = 12
    mic_phrase_time_limit: int = 7
    mic_ambient_noise_duration: float = 1.0
    use_whisper_stt: bool = True
    whisper_model: str = "small.en"
    whisper_record_seconds: int = 5
    whisper_short_reply_mode: bool = True
    whisper_use_vad: bool = False
    whisper_initial_prompt: str = "yes, no, sure, why not, please do, don't, not now, can you do it, tell me more, repeat that"

    @classmethod
    def from_env(cls, camera_index=None, mic_device_index=None):
        plantnet_api_key = os.getenv("PLANTNET_API_KEY", "")
        return cls(
            camera_index=camera_index if camera_index is not None else int(os.getenv("CAMERA_INDEX", "0")),
            use_clip=env_bool("USE_CLIP", True),
            use_llm_intent=env_bool("USE_LLM_INTENT", True),
            use_ollama=env_bool("USE_OLLAMA", True),
            dialogue_model=os.getenv("DIALOGUE_MODEL", os.getenv("MAIN_LLM_MODEL", "gemma3:1b")),
            vision_llm_model=os.getenv("VISION_LLM_MODEL", "gemma3:4b"),
            final_response_model=os.getenv("FINAL_RESPONSE_MODEL", "gemma3:4b"),
            main_llm_model=os.getenv("MAIN_LLM_MODEL", os.getenv("DIALOGUE_MODEL", "gemma3:1b")),
            ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
            ollama_dialogue_timeout=int(os.getenv("OLLAMA_DIALOGUE_TIMEOUT", "30")),
            ollama_text_timeout=int(os.getenv("OLLAMA_TEXT_TIMEOUT", "60")),
            ollama_image_timeout=int(os.getenv("OLLAMA_IMAGE_TIMEOUT", "180")),
            plant_id_provider=os.getenv("PLANT_ID_PROVIDER", "plantnet"),
            plantnet_api_key=plantnet_api_key,
            plantnet_project=os.getenv("PLANTNET_PROJECT", "all"),
            plantnet_lang=os.getenv("PLANTNET_LANG", "en"),
            plantnet_organ=os.getenv("PLANTNET_ORGAN", "auto"),
            plantnet_min_confidence=float(os.getenv("PLANTNET_MIN_CONFIDENCE", "0.45")),
            plantnet_high_confidence=float(os.getenv("PLANTNET_HIGH_CONFIDENCE", "0.70")),
            use_plant_id=env_bool("USE_PLANT_ID", bool(plantnet_api_key)),
            mic_device_index=mic_device_index if mic_device_index is not None else int(os.getenv("MIC_DEVICE_INDEX", "0")),
            mic_listen_timeout=int(os.getenv("MIC_LISTEN_TIMEOUT", "12")),
            mic_phrase_time_limit=int(os.getenv("MIC_PHRASE_TIME_LIMIT", "7")),
            mic_ambient_noise_duration=float(os.getenv("MIC_AMBIENT_NOISE_DURATION", "1")),
            use_whisper_stt=env_bool("USE_WHISPER_STT", True),
            whisper_model=os.getenv("WHISPER_MODEL", "small.en"),
            whisper_record_seconds=int(os.getenv("WHISPER_RECORD_SECONDS", "5")),
            whisper_short_reply_mode=env_bool("WHISPER_SHORT_REPLY_MODE", True),
            whisper_use_vad=env_bool("WHISPER_USE_VAD", False),
            whisper_initial_prompt=os.getenv(
                "WHISPER_INITIAL_PROMPT",
                "yes, no, sure, why not, please do, don't, not now, can you do it, tell me more, repeat that",
            ),
        )
