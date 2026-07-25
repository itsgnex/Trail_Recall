from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()

EXPECTED_MAC_IP = "10.117.240.212"
EXPECTED_PHONE_IP = "10.117.240.233"
MENTRA_STREAM_PATH = "live/mentra-live"
DEFAULT_ANDROID_TRAIL_URL = f"http://{EXPECTED_PHONE_IP}:8766"
DEFAULT_MENTRA_RTMP_URL = f"rtmp://{EXPECTED_MAC_IP}:1935/{MENTRA_STREAM_PATH}"
DEFAULT_MENTRA_RTSP_URL = f"rtsp://127.0.0.1:8554/{MENTRA_STREAM_PATH}"
DEFAULT_MENTRA_HLS_URL = f"http://{EXPECTED_MAC_IP}:8888/{MENTRA_STREAM_PATH}/index.m3u8"


def env_bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def env_list(name, default):
    value = os.getenv(name)
    if not value:
        return tuple(default)
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    focus_fraction: float = 0.45
    analysis_interval: float = 1.0
    dwell_seconds: float = 2.0
    cooldown_seconds: float = 9.0
    use_clip: bool = True
    use_llm_intent: bool = True
    use_ollama: bool = True
    dialogue_model: str = "gemma3:1b"
    vision_llm_model: str = "gemma3:1b"
    final_response_model: str = "gemma3:1b"
    main_llm_model: str = "gemma3:1b"
    ollama_model: str = "gemma3:1b"
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_dialogue_timeout: int = 12
    ollama_text_timeout: int = 20
    ollama_image_timeout: int = 45
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-3.1-flash-lite"
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_timeout: int = 8
    use_local_vision_llm: bool = False
    plant_id_provider: str = "plantnet"
    plantnet_api_key: str = ""
    plantnet_project: str = "all"
    plantnet_lang: str = "en"
    plantnet_organ: str = "auto"
    plantnet_min_confidence: float = 0.25
    plantnet_high_confidence: float = 0.70
    plant_response_ai: bool = True
    verify_plant_before_plantnet: bool = True
    plant_verify_min_confidence: float = 0.70
    use_plant_id: bool = False
    mic_device_index: int = 0
    mic_listen_timeout: int = 12
    mic_phrase_time_limit: int = 7
    mic_ambient_noise_duration: float = 1.0
    mic_ambient_refresh_duration: float = 0.25
    mic_post_tts_delay: float = 0.45
    mic_phrase_start_pause: float = 0.2
    use_whisper_stt: bool = True
    whisper_model: str = "small.en"
    wake_whisper_model: str = "tiny.en"
    wake_whisper_cpu_threads: int = 4
    wake_transcription_timeout_seconds: float = 2.0
    wake_min_rms: int = 140
    wake_ambient_multiplier: float = 2.5
    wake_min_voiced_ms: int = 250
    whisper_record_seconds: int = 5
    whisper_short_reply_mode: bool = True
    whisper_use_vad: bool = False
    whisper_initial_prompt: str = "yes, no, sure, why not, please do, don't, not now, can you do it, tell me more, repeat that"
    stt_provider: str = "openrouter_first"
    openai_api_key: str = ""
    openai_stt_model: str = "gpt-4o-mini-transcribe"
    openai_stt_url: str = "https://api.openai.com/v1/audio/transcriptions"
    openai_stt_timeout: int = 8
    openrouter_stt_model: str = "openai/gpt-4o-mini-transcribe"
    openrouter_stt_timeout_seconds: float = 3.0
    local_stt_timeout_seconds: float = 2.0
    reply_min_rms: int = 120
    reply_min_voiced_ms: int = 150
    follow_up_mode: bool = True
    follow_up_timeout_seconds: float = 1.5
    follow_up_silence_ms: int = 300
    confirmation_record_seconds: float = 2.0
    confirmation_silence_ms: int = 350
    max_follow_up_turns: int = 2
    speak_follow_up_offer: bool = False
    follow_up_silence_returns_to_scan: bool = True
    visual_prompt_cooldown_seconds: float = 7.0
    visual_reply_max_retries: int = 1
    visual_object_gone_seconds: float = 3.0
    visual_object_gone_min_frames: int = 8
    visual_rearm_scene_change_threshold: float = 18.0
    voice_activation_mode: bool = True
    wake_mode: bool = True
    wake_phrases: tuple[str, ...] = (
        "hey look",
        "okay look",
        "hey trail",
        "okay trail",
        "trail",
        "hey glasses",
        "hey assistant",
        "hey nova",
    )
    wake_listen_seconds: float = 3.0
    wake_capture_seconds: float = 2.0
    wake_silence_ms: int = 400
    command_record_seconds: float = 2.5
    speech_end_silence_ms: int = 450
    wake_cooldown_seconds: int = 2
    pause_wake_during_tts: bool = True
    allow_single_word_wake: bool = False
    wake_debug_transcripts: bool = True
    wake_log_empty_transcripts: bool = False
    wake_min_transcript_length: int = 2
    voice_command_mode: bool = True
    allow_general_questions: bool = True
    general_question_model: str = "gemma3:1b"
    image_task_model: str = "gemma3:1b"
    sign_memory_enabled: bool = True
    sign_memory_max_items: int = 30
    sign_memory_ttl_seconds: int = 600
    sign_clip_duplicate_threshold: float = 0.88
    sign_clip_possible_duplicate_threshold: float = 0.82
    sign_gemma_verify_duplicates: bool = True
    sign_duplicate_suppress_seconds: int = 180
    scene_memory_enabled: bool = True
    scene_memory_ttl_seconds: int = 3600
    scene_memory_max_items: int = 120
    android_trail_base_url: str = ""
    use_glasses_mic: bool = False
    latency_warn_threshold_ms: int = 1000
    latency_log_file_enabled: bool = False
    log_level: str = "INFO"
    latency_console_mode: str = "summary"
    vision_log_interval_seconds: float = 10.0
    mic_log_interval_seconds: float = 10.0
    terminal_compact_mode: bool = False

    @classmethod
    def from_env(cls, camera_index=None, mic_device_index=None):
        voice_profile = os.getenv("VOICE_PROFILE", "balanced").strip().lower()
        profile_defaults = {
            "fast": {
                "whisper_model": "small.en",
                "whisper_record_seconds": "3",
                "whisper_short_reply_mode": "false",
                "whisper_use_vad": "false",
                "wake_listen_seconds": "3.5",
                "command_record_seconds": "3",
                "wake_cooldown_seconds": "1",
                "mic_listen_timeout": "8",
                "mic_ambient_noise_duration": "0.75",
                "wake_log_empty_transcripts": "true",
            },
            "whisper_first": {
                "whisper_model": "small.en",
                "whisper_record_seconds": "4",
                "whisper_short_reply_mode": "false",
                "whisper_use_vad": "false",
                "wake_listen_seconds": "2.0",
                "command_record_seconds": "3",
                "wake_cooldown_seconds": "2",
                "mic_listen_timeout": "10",
                "mic_ambient_noise_duration": "1.0",
            },
            "balanced": {},
            "accurate": {
                "whisper_model": "medium.en",
                "whisper_record_seconds": "6",
                "whisper_short_reply_mode": "false",
                "whisper_use_vad": "true",
                "wake_listen_seconds": "3.5",
                "command_record_seconds": "3",
                "wake_cooldown_seconds": "1",
                "mic_listen_timeout": "14",
                "mic_ambient_noise_duration": "1.0",
            },
        }.get(voice_profile, {})

        def profile_default(name, default):
            return profile_defaults.get(name, default)

        plantnet_api_key = os.getenv("PLANTNET_API_KEY", "")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        stt_provider = os.getenv("STT_PROVIDER", "openrouter_first").strip().lower()
        if stt_provider == "openai" and not openai_api_key and openrouter_api_key:
            stt_provider = "openrouter_first"
        return cls(
            camera_index=camera_index if camera_index is not None else int(os.getenv("CAMERA_INDEX", "0")),
            use_clip=env_bool("USE_CLIP", True),
            use_llm_intent=env_bool("USE_LLM_INTENT", True),
            use_ollama=env_bool("USE_OLLAMA", True),
            dialogue_model=os.getenv("DIALOGUE_MODEL", os.getenv("MAIN_LLM_MODEL", "gemma3:1b")),
            vision_llm_model=os.getenv("VISION_LLM_MODEL", "gemma3:1b"),
            final_response_model=os.getenv("FINAL_RESPONSE_MODEL", "gemma3:1b"),
            main_llm_model=os.getenv("MAIN_LLM_MODEL", os.getenv("DIALOGUE_MODEL", "gemma3:1b")),
            ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:1b"),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
            ollama_dialogue_timeout=int(os.getenv("OLLAMA_DIALOGUE_TIMEOUT", "12")),
            ollama_text_timeout=int(os.getenv("OLLAMA_TEXT_TIMEOUT", "20")),
            ollama_image_timeout=int(os.getenv("OLLAMA_IMAGE_TIMEOUT", "45")),
            openrouter_api_key=openrouter_api_key,
            openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemini-3.1-flash-lite"),
            openrouter_url=os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"),
            openrouter_timeout=int(os.getenv("OPENROUTER_TIMEOUT", "8")),
            use_local_vision_llm=env_bool("USE_LOCAL_VISION_LLM", False),
            plant_id_provider=os.getenv("PLANT_ID_PROVIDER", "plantnet"),
            plantnet_api_key=plantnet_api_key,
            plantnet_project=os.getenv("PLANTNET_PROJECT", "all"),
            plantnet_lang=os.getenv("PLANTNET_LANG", "en"),
            plantnet_organ=os.getenv("PLANTNET_ORGAN", "auto"),
            plantnet_min_confidence=float(os.getenv("PLANTNET_MIN_CONFIDENCE", "0.25")),
            plantnet_high_confidence=float(os.getenv("PLANTNET_HIGH_CONFIDENCE", "0.70")),
            plant_response_ai=env_bool("PLANT_RESPONSE_AI", True),
            verify_plant_before_plantnet=env_bool("VERIFY_PLANT_BEFORE_PLANTNET", True),
            plant_verify_min_confidence=float(os.getenv("PLANT_VERIFY_MIN_CONFIDENCE", "0.70")),
            use_plant_id=env_bool("USE_PLANT_ID", bool(plantnet_api_key)),
            mic_device_index=mic_device_index if mic_device_index is not None else int(os.getenv("MIC_DEVICE_INDEX", "0")),
            mic_listen_timeout=int(os.getenv("MIC_LISTEN_TIMEOUT", profile_default("mic_listen_timeout", "12"))),
            mic_phrase_time_limit=int(os.getenv("MIC_PHRASE_TIME_LIMIT", "7")),
            mic_ambient_noise_duration=float(os.getenv("MIC_AMBIENT_NOISE_DURATION", profile_default("mic_ambient_noise_duration", "1"))),
            mic_ambient_refresh_duration=float(os.getenv("MIC_AMBIENT_REFRESH_DURATION", "0.25")),
            mic_post_tts_delay=float(os.getenv("MIC_POST_TTS_DELAY", "0.45")),
            mic_phrase_start_pause=float(os.getenv("MIC_PHRASE_START_PAUSE", "0.2")),
            use_whisper_stt=env_bool("USE_WHISPER_STT", True),
            whisper_model=os.getenv("WHISPER_MODEL", profile_default("whisper_model", "small.en")),
            wake_whisper_model=os.getenv("WAKE_WHISPER_MODEL", "tiny.en"),
            wake_whisper_cpu_threads=int(os.getenv("WAKE_WHISPER_CPU_THREADS", "4")),
            wake_transcription_timeout_seconds=float(os.getenv("WAKE_TRANSCRIPTION_TIMEOUT_SECONDS", "2")),
            wake_min_rms=int(os.getenv("WAKE_MIN_RMS", "140")),
            wake_ambient_multiplier=float(os.getenv("WAKE_AMBIENT_MULTIPLIER", "2.5")),
            wake_min_voiced_ms=int(os.getenv("WAKE_MIN_VOICED_MS", "250")),
            whisper_record_seconds=int(os.getenv("WHISPER_RECORD_SECONDS", profile_default("whisper_record_seconds", "5"))),
            whisper_short_reply_mode=env_bool("WHISPER_SHORT_REPLY_MODE", profile_default("whisper_short_reply_mode", "true")),
            whisper_use_vad=env_bool("WHISPER_USE_VAD", profile_default("whisper_use_vad", "false")),
            whisper_initial_prompt=os.getenv(
                "WHISPER_INITIAL_PROMPT",
                "yes, no, sure, why not, please do, don't, not now, can you do it, tell me more, repeat that",
            ),
            stt_provider=stt_provider,
            openai_api_key=openai_api_key,
            openai_stt_model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
            openai_stt_url=os.getenv("OPENAI_STT_URL", "https://api.openai.com/v1/audio/transcriptions"),
            openai_stt_timeout=int(os.getenv("OPENAI_STT_TIMEOUT", "8")),
            openrouter_stt_model=os.getenv("OPENROUTER_STT_MODEL", "openai/gpt-4o-mini-transcribe"),
            openrouter_stt_timeout_seconds=float(os.getenv("OPENROUTER_STT_TIMEOUT_SECONDS", "3")),
            local_stt_timeout_seconds=float(os.getenv("LOCAL_STT_TIMEOUT_SECONDS", "2")),
            reply_min_rms=int(os.getenv("REPLY_MIN_RMS", "120")),
            reply_min_voiced_ms=int(os.getenv("REPLY_MIN_VOICED_MS", "150")),
            follow_up_mode=env_bool("FOLLOW_UP_MODE", True),
            follow_up_timeout_seconds=float(os.getenv("FOLLOW_UP_RECORD_SECONDS", os.getenv("FOLLOW_UP_TIMEOUT_SECONDS", "1.5"))),
            follow_up_silence_ms=int(os.getenv("FOLLOW_UP_SILENCE_MS", "300")),
            confirmation_record_seconds=float(os.getenv("CONFIRMATION_RECORD_SECONDS", "2")),
            confirmation_silence_ms=int(os.getenv("CONFIRMATION_SILENCE_MS", "350")),
            max_follow_up_turns=int(os.getenv("MAX_FOLLOW_UP_TURNS", "2")),
            speak_follow_up_offer=env_bool("SPEAK_FOLLOW_UP_OFFER", False),
            follow_up_silence_returns_to_scan=env_bool("FOLLOW_UP_SILENCE_RETURNS_TO_SCAN", True),
            visual_prompt_cooldown_seconds=float(os.getenv("VISUAL_PROMPT_COOLDOWN_SECONDS", "7")),
            visual_reply_max_retries=int(os.getenv("VISUAL_REPLY_MAX_RETRIES", "1")),
            visual_object_gone_seconds=float(os.getenv("VISUAL_OBJECT_GONE_SECONDS", "3")),
            visual_object_gone_min_frames=int(os.getenv("VISUAL_OBJECT_GONE_MIN_FRAMES", "8")),
            visual_rearm_scene_change_threshold=float(os.getenv("VISUAL_REARM_SCENE_CHANGE_THRESHOLD", "18")),
            voice_activation_mode=env_bool("VOICE_ACTIVATION_MODE", True),
            wake_mode=env_bool("WAKE_MODE", True),
            wake_phrases=env_list(
                "WAKE_PHRASES",
                (
                    "hey look",
                    "okay look",
                    "hey trail",
                    "okay trail",
                    "trail",
                    "hey glasses",
                    "hey assistant",
                    "hey nova",
                ),
            ),
            wake_listen_seconds=float(os.getenv("WAKE_LISTEN_SECONDS", profile_default("wake_listen_seconds", "3.0"))),
            wake_capture_seconds=float(os.getenv("WAKE_CAPTURE_SECONDS", "2.0")),
            wake_silence_ms=int(os.getenv("WAKE_SILENCE_MS", "400")),
            command_record_seconds=float(os.getenv("COMMAND_RECORD_SECONDS", profile_default("command_record_seconds", "2.5"))),
            speech_end_silence_ms=int(os.getenv("SPEECH_END_SILENCE_MS", "450")),
            wake_cooldown_seconds=int(os.getenv("WAKE_COOLDOWN_SECONDS", profile_default("wake_cooldown_seconds", "2"))),
            pause_wake_during_tts=env_bool("PAUSE_WAKE_DURING_TTS", True),
            allow_single_word_wake=env_bool("ALLOW_SINGLE_WORD_WAKE", False),
            wake_debug_transcripts=env_bool("WAKE_DEBUG_TRANSCRIPTS", True),
            wake_log_empty_transcripts=env_bool("WAKE_LOG_EMPTY_TRANSCRIPTS", profile_default("wake_log_empty_transcripts", "false")),
            wake_min_transcript_length=int(os.getenv("WAKE_MIN_TRANSCRIPT_LENGTH", "2")),
            voice_command_mode=env_bool("VOICE_COMMAND_MODE", True),
            allow_general_questions=env_bool("ALLOW_GENERAL_QUESTIONS", True),
            general_question_model=os.getenv("GENERAL_QUESTION_MODEL", "gemma3:1b"),
            image_task_model=os.getenv("IMAGE_TASK_MODEL", "gemma3:1b"),
            sign_memory_enabled=env_bool("SIGN_MEMORY_ENABLED", True),
            sign_memory_max_items=int(os.getenv("SIGN_MEMORY_MAX_ITEMS", "30")),
            sign_memory_ttl_seconds=int(os.getenv("SIGN_MEMORY_TTL_SECONDS", "600")),
            sign_clip_duplicate_threshold=float(os.getenv("SIGN_CLIP_DUPLICATE_THRESHOLD", "0.88")),
            sign_clip_possible_duplicate_threshold=float(os.getenv("SIGN_CLIP_POSSIBLE_DUPLICATE_THRESHOLD", "0.82")),
            sign_gemma_verify_duplicates=env_bool("SIGN_GEMMA_VERIFY_DUPLICATES", True),
            sign_duplicate_suppress_seconds=int(os.getenv("SIGN_DUPLICATE_SUPPRESS_SECONDS", "180")),
            scene_memory_enabled=env_bool("SCENE_MEMORY_ENABLED", True),
            scene_memory_ttl_seconds=int(os.getenv("SCENE_MEMORY_TTL_SECONDS", "3600")),
            scene_memory_max_items=int(os.getenv("SCENE_MEMORY_MAX_ITEMS", "120")),
            android_trail_base_url=os.getenv("ANDROID_TRAIL_URL", "").strip(),
            use_glasses_mic=env_bool("USE_GLASSES_MIC", True),
            latency_warn_threshold_ms=int(os.getenv("LATENCY_WARN_THRESHOLD_MS", "1000")),
            latency_log_file_enabled=env_bool("LATENCY_LOG_FILE_ENABLED", False),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            latency_console_mode=os.getenv("LATENCY_CONSOLE_MODE", "summary").strip().lower(),
            vision_log_interval_seconds=float(os.getenv("VISION_LOG_INTERVAL_SECONDS", "10")),
            mic_log_interval_seconds=float(os.getenv("MIC_LOG_INTERVAL_SECONDS", "10")),
            terminal_compact_mode=env_bool("TERMINAL_COMPACT_MODE", False),
        )
