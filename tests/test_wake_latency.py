import io
import inspect
import json
import os
import sys
import threading
import time
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.config import Config
from src.common_tts import lookup_by_text
from src.audio_http import prepare_glasses_speech, _BackendHandler
from src.inference import local_model, visual_inference_allowed
from src.latency import log_stage, new_interaction, set_interaction, summarize
from src.intent import Intent
from src.session_state import SessionState
import main
import src.speech_in as speech_in
import src.speech_out as speech_out


class FakeAudio:
    def __init__(self, pcm_value=500, chunks=6):
        self.pcm = pcm_chunk(pcm_value) * chunks

    def get_wav_data(self, convert_rate=None, convert_width=None):
        return b"RIFF....WAVEfmt "

    def get_raw_data(self, convert_rate=None, convert_width=None):
        return self.pcm


class FakeSegment:
    text = "hey trail"


class FakeWhisperModel:
    loaded = []
    active = 0
    max_active = 0
    sleep_seconds = 0

    def __init__(self, name, device="auto", compute_type="int8", **kwargs):
        self.name = name
        self.kwargs = kwargs
        FakeWhisperModel.loaded.append(name)

    def transcribe(self, *_args, **_kwargs):
        FakeWhisperModel.active += 1
        FakeWhisperModel.max_active = max(FakeWhisperModel.max_active, FakeWhisperModel.active)
        if FakeWhisperModel.sleep_seconds:
            time.sleep(FakeWhisperModel.sleep_seconds)
        FakeWhisperModel.active -= 1
        return [FakeSegment()], None


class FakeResponse:
    status_code = 400

    def json(self):
        return {"error": "bad"}


def fake_faster_whisper():
    return {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeWhisperModel)}


class FakeGlassesBuffer:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def clear(self):
        pass

    def read_bytes(self, _need, timeout=None):
        if self.chunks:
            return self.chunks.pop(0)
        return b""


def pcm_chunk(value, samples=800):
    return int(value).to_bytes(2, "little", signed=True) * samples


class WakeLatencyTests(unittest.TestCase):
    def setUp(self):
        new_interaction("test")
        FakeWhisperModel.loaded = []
        FakeWhisperModel.active = 0
        FakeWhisperModel.max_active = 0
        FakeWhisperModel.sleep_seconds = 0
        speech_in._whisper_model = None
        speech_in._whisper_name = None
        speech_in._wake_whisper_model = None
        speech_in._wake_whisper_name = None

    def test_tiny_used_only_for_wake_preload(self):
        with patch.dict(sys.modules, fake_faster_whisper()):
            speech_in.preload_whisper_model(Config(wake_whisper_model="tiny.en", whisper_model="small.en"))
        self.assertEqual(FakeWhisperModel.loaded, ["tiny.en"])
        self.assertEqual(speech_in._wake_whisper_model.kwargs["cpu_threads"], 4)
        self.assertEqual(speech_in._wake_whisper_model.kwargs["num_workers"], 1)

    @patch("requests.post", return_value=FakeResponse())
    def test_small_loaded_lazily_after_api_failure(self, _post):
        with patch("src.speech_in._transcribe_with_whisper_process", return_value="hey trail"):
            text = speech_in._transcribe_primary(FakeAudio(), Config(openrouter_api_key="present", whisper_model="small.en"))
        self.assertEqual(text, "hey trail")

    def test_wake_stt_pauses_heavy_visual_inference(self):
        self.assertTrue(visual_inference_allowed())
        with local_model("wake"):
            self.assertFalse(visual_inference_allowed())
        self.assertTrue(visual_inference_allowed())

    def test_no_concurrent_whisper_calls(self):
        active = 0
        max_active = 0

        def slow_worker(*_args):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            time.sleep(0.03)
            active -= 1
            return "ok"

        audio = FakeAudio()
        config = Config(whisper_model="small.en")
        with patch("src.speech_in._transcribe_with_whisper_process", side_effect=slow_worker):
            threads = [threading.Thread(target=speech_in.transcribe_with_whisper, args=(audio, config, True)) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(max_active, 1)

    def test_repeated_hey_transcript_rejected_without_full_print(self):
        text = " ".join(["hey"] * 40)
        rejection = speech_in.wake_transcript_rejection(text)
        self.assertIsNotNone(rejection)
        self.assertLess(len(rejection["preview"]), len(text))

    def test_repeated_look_transcript_rejected(self):
        self.assertIsNotNone(speech_in.wake_transcript_rejection(" ".join(["look"] * 30)))

    def test_valid_wake_command_not_rejected(self):
        self.assertIsNone(speech_in.wake_transcript_rejection("hey trail can you hear me"))

    def test_single_word_replies_not_repetition_hallucinations(self):
        self.assertIsNone(speech_in.wake_transcript_rejection("no"))
        self.assertIsNone(speech_in.wake_transcript_rejection("yes"))

    def test_wake_requires_prefix(self):
        config = Config()
        self.assertFalse(speech_in.wake_phrase_detected("can you hear me", config))
        self.assertFalse(speech_in.wake_phrase_detected("what is this", config))
        self.assertFalse(speech_in.wake_phrase_detected("look at this", config))

    def test_trail_prefix_wake_extracts_command(self):
        config = Config()
        self.assertTrue(speech_in.wake_phrase_detected("hey trail what is this", config))
        self.assertTrue(speech_in.wake_phrase_detected("trail what is this", config))
        self.assertEqual(speech_in.strip_wake_phrase("hey trail what is this", config.wake_phrases, False), "what is this")
        self.assertEqual(speech_in.strip_wake_phrase("trail what is this", config.wake_phrases, False), "what is this")

    def test_wake_only_triggers_fresh_command_capture(self):
        self.assertEqual(speech_in.strip_wake_phrase("hey trail", ("hey trail",), False), "")
        self.assertTrue(speech_in.is_incomplete_command("hey hey hey"))

    def test_zero_audio_excluded_from_ambient_calibration(self):
        chunks = [pcm_chunk(0) for _ in range(10)]
        speech_in._wake_ambient_rms = 77
        with patch("src.mic_ingest.glasses_mic_buffer", return_value=FakeGlassesBuffer(chunks)):
            value = speech_in.calibrate_wake_ambient(Config(use_glasses_mic=True), seconds=0.1)
        self.assertEqual(value, 77)

    def test_low_rms_background_audio_is_rejected(self):
        chunks = [pcm_chunk(70) for _ in range(20)]
        with patch("src.mic_ingest.glasses_mic_buffer", return_value=FakeGlassesBuffer(chunks)):
            audio = speech_in._record_glasses_audio(Config(use_glasses_mic=True), 1.0, quiet=True, latency_prefix="WAKE_AUDIO_CAPTURE")
        self.assertEqual(audio, "")

    def test_one_noisy_frame_does_not_start_speech(self):
        chunks = [pcm_chunk(0) for _ in range(3)] + [pcm_chunk(200)] + [pcm_chunk(0) for _ in range(12)]
        with patch("src.mic_ingest.glasses_mic_buffer", return_value=FakeGlassesBuffer(chunks)):
            audio = speech_in._record_glasses_audio(Config(use_glasses_mic=True), 1.0, quiet=True, latency_prefix="WAKE_AUDIO_CAPTURE")
        self.assertEqual(audio, "")

    def test_real_hey_trail_audio_passes_dynamic_threshold(self):
        chunks = [pcm_chunk(0) for _ in range(4)] + [pcm_chunk(220) for _ in range(6)] + [pcm_chunk(0) for _ in range(8)]
        with patch("src.mic_ingest.glasses_mic_buffer", return_value=FakeGlassesBuffer(chunks)):
            audio = speech_in._record_glasses_audio(Config(use_glasses_mic=True), 1.5, quiet=True, silence_ms=300, latency_prefix="WAKE_AUDIO_CAPTURE")
        self.assertFalse(isinstance(audio, str))

    def test_wake_transcription_timeout_recovery(self):
        FakeWhisperModel.sleep_seconds = 0.05
        with patch.dict(sys.modules, fake_faster_whisper()):
            text = speech_in.transcribe_wake_with_whisper(FakeAudio(), Config(wake_transcription_timeout_seconds=0.01))
        self.assertIsNone(text)

    def test_low_rms_reply_skips_openrouter_and_local_fallback(self):
        with patch("src.speech_in.transcribe_with_openrouter") as openrouter, patch("src.speech_in.transcribe_with_whisper") as whisper:
            text = speech_in._transcribe_with_fallback(FakeAudio(pcm_value=50), Config(), True, False, "recording command")
        self.assertEqual(text, "")
        openrouter.assert_not_called()
        whisper.assert_not_called()

    def test_empty_openrouter_low_quality_skips_small(self):
        with patch("src.speech_in.transcribe_with_openrouter", return_value="") as openrouter, patch("src.speech_in.transcribe_with_whisper") as whisper:
            text = speech_in._transcribe_with_fallback(FakeAudio(pcm_value=60), Config(), True, False, "recording command")
        self.assertEqual(text, "")
        openrouter.assert_not_called()
        whisper.assert_not_called()

    def test_local_fallback_forcibly_stops_within_timeout(self):
        def slow_worker(*_args):
            time.sleep(5)

        started = time.monotonic()
        with patch("src.speech_in._whisper_process_worker", side_effect=slow_worker):
            text = speech_in._transcribe_with_whisper_process(FakeAudio(), Config(local_stt_timeout_seconds=0.05), 0.05)
        self.assertEqual(text, "")
        self.assertLess(time.monotonic() - started, 1.0)

    def test_reply_scheduled_after_wav_duration(self):
        event_id = "evt-playback"
        speech_out._begin_playback(event_id, "hello world", False)
        speech_out.set_playback_duration(event_id, 300)
        buf = io.StringIO()
        with redirect_stdout(buf):
            speech_out.notify_playback_started(event_id)
        self.assertIn("REPLY_CAPTURE scheduledInMs=", buf.getvalue())
        self.assertIn("audioDurationMs=300", buf.getvalue())

    def test_interaction_id_stays_consistent(self):
        set_interaction("abc123")
        first = log_stage("WAKE_LISTEN_START")
        second = log_stage("WAKE_AUDIO_CAPTURE_END")
        self.assertEqual(first["interactionId"], "abc123")
        self.assertEqual(second["interactionId"], "abc123")

    def test_original_interaction_id_survives_http_playback_callbacks(self):
        set_interaction("orig123")
        from src.latency import set_event

        set_event("evt123")
        items = []

        def worker():
            items.append(log_stage("BLUETOOTH_PLAYBACK_START", event_id="evt123"))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertEqual(items[0]["interactionId"], "orig123")

    def test_only_actual_callback_labeled_bluetooth_playback_start(self):
        source = inspect.getsource(_BackendHandler._send_wav)
        self.assertIn("AUDIO_READY_FOR_ANDROID", source)
        self.assertNotIn("BLUETOOTH_PLAYBACK_START", source)

    def test_stage_durations_use_monotonic_timing(self):
        with patch("src.latency.time.monotonic", side_effect=[10.0, 10.1, 10.4]):
            new_interaction("mono")
            log_stage("WAKE_LISTEN_START")
            item = log_stage("WAKE_AUDIO_CAPTURE_END")
        self.assertEqual(item["elapsedMs"], 300)
        self.assertEqual(item["totalMs"], 400)

    def test_info_mode_suppresses_latency_blocks(self):
        buf = io.StringIO()
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LATENCY_CONSOLE_MODE": "summary"}), redirect_stdout(buf):
            set_interaction("quiet")
            log_stage("WAKE_LISTEN_START")
        self.assertNotIn("LATENCY\n", buf.getvalue())

    def test_expected_capture_duration_does_not_warn_at_info(self):
        buf = io.StringIO()
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LATENCY_CONSOLE_MODE": "summary", "LATENCY_WARN_THRESHOLD_MS": "1000"}), patch(
            "src.latency.time.monotonic", side_effect=[1.0, 1.0, 2.5]
        ), redirect_stdout(buf):
            new_interaction("capture")
            log_stage("COMMAND_CAPTURE_START")
            log_stage("COMMAND_CAPTURE_END")
        self.assertNotIn("LATENCY_WARNING", buf.getvalue())

    def test_info_compact_suppresses_center_crop_and_http_access(self):
        self.assertIn("app_log.debug(f\"center crop:", inspect.getsource(main))
        buf = io.StringIO()
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "TERMINAL_COMPACT_MODE": "1"}), redirect_stdout(buf):
            _BackendHandler.log_message(object(), '"GET /command HTTP/1.1" 200 -')
        self.assertNotIn("backend-http:", buf.getvalue())

    def test_debug_mode_retains_latency_blocks(self):
        buf = io.StringIO()
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}), redirect_stdout(buf):
            set_interaction("debug")
            log_stage("WAKE_LISTEN_START")
        self.assertIn("LATENCY\n", buf.getvalue())

    def test_terminal_transcript_truncation(self):
        self.assertTrue(speech_in._preview("x" * 200).endswith("..."))

    def test_compact_latency_summary_format(self):
        set_interaction("sum2", started_at=1.0, stages=[
            {"stage": "WAKE_LISTEN_START", "totalMs": 0, "elapsedMs": 0, "interactionId": "sum2"},
            {"stage": "WAKE_AUDIO_CAPTURE_END", "totalMs": 100, "elapsedMs": 100, "interactionId": "sum2"},
        ])
        buf = io.StringIO()
        with redirect_stdout(buf):
            summarize()
        self.assertIn("LATENCY_SUMMARY interactionId=sum2", buf.getvalue())

    def test_latency_summary_identifies_slowest_stage(self):
        set_interaction("sum1", started_at=1.0, stages=[
            {"stage": "WAKE_LISTEN_START", "totalMs": 0, "elapsedMs": 0, "interactionId": "sum1"},
            {"stage": "WAKE_AUDIO_CAPTURE_END", "totalMs": 200, "elapsedMs": 200, "interactionId": "sum1"},
            {"stage": "TRANSCRIPT_READY", "totalMs": 900, "elapsedMs": 700, "interactionId": "sum1"},
        ])
        out = summarize()
        self.assertEqual(out["slowestStage"], "TRANSCRIPT_READY")
        self.assertEqual(out["slowestStageMs"], 700)

    def test_jsonl_output_is_valid_json(self):
        path = Path("logs/latency.jsonl")
        if path.exists():
            path.unlink()
        with patch.dict(os.environ, {"LATENCY_LOG_FILE_ENABLED": "1"}):
            set_interaction("json1")
            log_stage("WAKE_LISTEN_START")
        self.assertTrue(path.exists())
        json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])

    def test_mic_and_vision_rate_limiting_helpers(self):
        from src import app_log

        with patch("src.app_log.time.monotonic", side_effect=[0, 1, 11]):
            self.assertTrue(app_log.rate_limited("k", 10, "one"))
            self.assertFalse(app_log.rate_limited("k", 10, "two"))
            self.assertTrue(app_log.rate_limited("k", 10, "three"))

    def test_api_keys_never_appear_in_logs(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            set_interaction("secret")
            log_stage("OPENROUTER_STT_START", api_key="sk-secret", authorization="Bearer sk-secret", model="x")
        self.assertNotIn("sk-secret", buf.getvalue())

    def test_common_audio_hit_for_new_phrase(self):
        entry = lookup_by_text("Yes, I’m listening. What would you like me to check?")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["phraseId"], "general_yes_listening_check")

    def test_common_audio_hit_for_added_exact_sentence(self):
        text = "I can hear you. Please tell me what you want me to look at."
        entry = lookup_by_text(text)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["phraseId"], "general_can_hear_tell_look")
        record = prepare_glasses_speech(text, "evt-common-test")
        self.assertEqual(record["phraseId"], "general_can_hear_tell_look")
        self.assertEqual(record["generationMs"], 0)

    def test_common_audio_hit_for_three_new_phrases(self):
        expected = {
            "Yes?": "general_yes_short",
            "I’m sorry, I wasn’t sure what you wanted. Would you like me to help with this?": "general_unsure_help_this",
            "I didn’t catch that. I’ll go back to watching quietly.": "general_didnt_catch_watch_quietly",
            "I can see something in the center, but it is not clear enough yet. Try holding it steadier or moving a little closer.": "general_center_not_clear_move_closer",
        }
        for text, phrase_id in expected.items():
            entry = lookup_by_text(text)
            self.assertIsNotNone(entry)
            self.assertEqual(entry["phraseId"], phrase_id)

    def test_successful_openrouter_command_survives_routing(self):
        state = SessionState()
        buf = io.StringIO()
        with patch("main.speak"), patch("main.listen", return_value="what is this"), patch(
            "main.classify_intent_with_source", return_value=(Intent.CANCEL, "test")
        ), redirect_stdout(buf):
            main.handle_voice_command("hey trail", Config(), state, camera=None)
        self.assertIn('COMMAND transcript="what is this"', buf.getvalue())
        self.assertEqual(state.last_user_transcript, "what is this")

    def test_assistant_echo_rejected_and_recaptured_once(self):
        speech_out._last_spoken_text = "I did not catch the full question. Please say it again."
        speech_out._last_spoken_at = time.monotonic()
        with patch("src.speech_in._record_audio", side_effect=[FakeAudio(), FakeAudio()]) as record, patch(
            "src.speech_in._transcribe_with_fallback",
            side_effect=["It said it did not catch the full question.", "no"],
        ):
            text = speech_in.listen(Config(), typed_fallback=False, record_seconds=1, label="recording command")
        self.assertEqual(text, "no")
        self.assertEqual(record.call_count, 2)


if __name__ == "__main__":
    unittest.main()
