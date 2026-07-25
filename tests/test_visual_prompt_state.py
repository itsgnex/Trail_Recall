import unittest
from unittest.mock import patch

import numpy as np

from src.config import Config
from src.intent import Intent
from src.llm import plant_response_from_id
from src.plant_id import PlantIdResult
from src.session_state import SessionState
from src import speech_out
from src.camera import _OpenCvStreamCamera
from src.rtmp_audio_ingest import RtmpAudioIngest
import main


def crop(value):
    return np.full((80, 80, 3), value, dtype=np.uint8)


class VisualPromptStateTests(unittest.TestCase):
    def setUp(self):
        speech_out._speaking.clear()
        speech_out._active_playback = None

    def test_same_plant_across_100_frames_prompts_once(self):
        state = SessionState()
        config = Config(visual_prompt_cooldown_seconds=30)
        image = crop(80)
        self.assertTrue(main.visual_prompt_should_start("plant", image, config, state, now=1.0))
        main.visual_interaction_complete(state, config, "yes", now=2.0)
        for index in range(100):
            self.assertFalse(main.visual_prompt_should_start("plant", image, config, state, now=3.0 + index))

    def test_same_sign_across_consecutive_frames_prompts_once(self):
        state = SessionState()
        config = Config()
        image = crop(120)
        self.assertTrue(main.visual_prompt_should_start("sign", image, config, state, now=1.0))
        for index in range(10):
            self.assertFalse(main.visual_prompt_should_start("sign", image, config, state, now=1.1 + index * 0.1))

    def test_detection_during_tts_does_not_interrupt_speech(self):
        with patch("main.is_speaking", return_value=True):
            state = SessionState()
            self.assertFalse(main.visual_prompt_should_start("plant", crop(40), Config(), state, now=1.0))
            self.assertEqual(state.visual_prompt_state, main.VISUAL_IDLE)

    def test_detection_while_waiting_for_reply_is_suppressed(self):
        state = SessionState(visual_prompt_state=main.VISUAL_WAITING_FOR_REPLY)
        self.assertFalse(main.visual_prompt_should_start("plant", crop(50), Config(), state, now=1.0))

    def test_yes_produces_one_analysis_and_one_result(self):
        state = SessionState(visual_prompt_state=main.VISUAL_PROMPT_PLAYING, visual_interaction_id="plant-1")
        with patch("main.speak") as speak, patch("main.listen", return_value="yes") as listen, patch(
            "main.classify_intent_with_source", return_value=(Intent.EXPLAIN_PLANT, "test")
        ), patch("main.answer_for", return_value="plant result") as answer, patch("main.maybe_store_scene_memory"):
            main.handle_trigger("plant", crop(70), Config(), state, camera=None, crop_path="plant.jpg")
        self.assertEqual(answer.call_count, 1)
        self.assertEqual(speak.call_count, 2)
        self.assertEqual(state.visual_prompt_state, main.VISUAL_RESULT_PLAYING)
        self.assertTrue(listen.call_args.kwargs["after_tts"])
        speak.call_args.kwargs["on_complete"]("event", "done")
        self.assertEqual(state.visual_prompt_state, main.VISUAL_COOLDOWN)

    def test_no_ends_the_interaction(self):
        state = SessionState(visual_prompt_state=main.VISUAL_PROMPT_PLAYING)
        with patch("main.speak") as speak, patch("main.listen", return_value="no"), patch(
            "main.classify_intent_with_source", return_value=(Intent.CANCEL, "test")
        ), patch("main.answer_for") as answer:
            main.handle_trigger("plant", crop(70), Config(), state, camera=None, crop_path="plant.jpg")
        answer.assert_not_called()
        self.assertLessEqual(speak.call_count, 2)
        self.assertEqual(state.visual_prompt_state, main.VISUAL_RESULT_PLAYING)
        speak.call_args.kwargs["on_complete"]("event", "done")
        self.assertEqual(state.visual_prompt_state, main.VISUAL_COOLDOWN)

    def test_silence_does_not_trigger_another_immediate_question(self):
        state = SessionState(visual_prompt_state=main.VISUAL_PROMPT_PLAYING)
        config = Config(visual_prompt_cooldown_seconds=30)
        with patch("main.speak"), patch("main.listen", return_value=""):
            main.handle_trigger("plant", crop(70), config, state, camera=None, crop_path="plant.jpg")
        self.assertEqual(state.visual_prompt_state, main.VISUAL_COOLDOWN)
        self.assertFalse(main.visual_prompt_should_start("plant", crop(70), config, state, now=1.0))

    def test_clarification_happens_at_most_once(self):
        state = SessionState(visual_prompt_state=main.VISUAL_PROMPT_PLAYING)
        with patch("main.speak") as speak, patch("main.listen", return_value="maybe"), patch(
            "main.classify_intent_with_source", return_value=(Intent.ASK_CLARIFICATION, "test")
        ), patch("main.listen_for_follow_up_reply", return_value=("still maybe", Intent.ASK_CLARIFICATION, "test")) as follow:
            main.handle_trigger("plant", crop(70), Config(visual_reply_max_retries=1), state, camera=None, crop_path="plant.jpg")
        self.assertEqual(follow.call_count, 1)
        self.assertEqual(state.visual_prompt_state, main.VISUAL_COOLDOWN)
        self.assertLessEqual(speak.call_count, 2)

    def test_same_object_cannot_retrigger_during_cooldown(self):
        state = SessionState()
        config = Config(visual_prompt_cooldown_seconds=30)
        image = crop(90)
        self.assertTrue(main.visual_prompt_should_start("plant", image, config, state, now=1.0))
        main.visual_interaction_complete(state, config, "yes", now=2.0)
        self.assertFalse(main.visual_prompt_should_start("plant", image, config, state, now=10.0))

    def test_object_triggers_after_leaving_and_reappearing(self):
        state = SessionState()
        config = Config(visual_prompt_cooldown_seconds=5, visual_object_gone_seconds=3)
        image = crop(90)
        self.assertTrue(main.visual_prompt_should_start("plant", image, config, state, now=1.0))
        main.visual_interaction_complete(state, config, "yes", now=2.0)
        self.assertFalse(main.visual_prompt_should_start("plant", image, config, state, now=8.0))
        for index in range(8):
            main.visual_mark_absence("other", config, state, crop=crop(0), now=8.0 + index * 0.5)
        self.assertTrue(main.visual_prompt_should_start("plant", image, config, state, now=13.0))

    def test_intermitent_other_frames_do_not_mark_plant_gone(self):
        state = SessionState()
        config = Config(visual_object_gone_seconds=3, visual_object_gone_min_frames=8)
        image = crop(90)
        self.assertTrue(main.visual_prompt_should_start("plant", image, config, state, now=1.0))
        for index in range(7):
            main.visual_mark_absence("other", config, state, crop=crop(0), now=2.0 + index * 0.2)
        self.assertTrue(state.visual_object_present)

    def test_same_still_visible_plant_does_not_retrigger_after_seven_second_cooldown(self):
        state = SessionState()
        config = Config(visual_prompt_cooldown_seconds=7)
        image = crop(90)
        self.assertTrue(main.visual_prompt_should_start("plant", image, config, state, now=1.0))
        main.visual_interaction_complete(state, config, "yes", now=2.0)
        self.assertFalse(main.visual_prompt_should_start("plant", image, config, state, now=20.0))

    def test_final_plant_result_never_schedules_reply_capture(self):
        speech_out._begin_playback("plant-final", "plant result", False, expect_reply=False)
        speech_out.set_playback_duration("plant-final", 300)
        with patch("builtins.print") as printed:
            speech_out.notify_playback_started("plant-final")
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("REPLY_CAPTURE_SKIPPED reason=final_result", output)
        self.assertNotIn("REPLY_CAPTURE scheduledInMs", output)

    def test_final_sign_result_never_schedules_reply_capture(self):
        speech_out._begin_playback("sign-final", "sign result", False, expect_reply=False)
        speech_out.set_playback_duration("sign-final", 300)
        with patch("builtins.print") as printed:
            speech_out.notify_playback_started("sign-final")
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("REPLY_CAPTURE_SKIPPED reason=final_result", output)

    def test_question_prompt_still_schedules_reply_capture(self):
        speech_out._begin_playback("question", "Would you like help?", False, expect_reply=True)
        speech_out.set_playback_duration("question", 300)
        with patch("builtins.print") as printed:
            speech_out.notify_playback_started("question")
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("REPLY_CAPTURE scheduledInMs", output)

    def test_negative_reply_timing_cannot_print_started(self):
        speech_out._begin_playback("stale", "Would you like help?", False, expect_reply=True)
        with speech_out._playback_lock:
            speech_out._active_playback["replyCaptureExpectedAt"] = 10.0
        with patch("src.speech_out.time.monotonic", return_value=8.0), patch("builtins.print") as printed:
            speech_out._clear_tts_state("stale", "playback_timeout")
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("REPLY_CAPTURE_SKIPPED reason=stale_event", output)
        self.assertNotIn("REPLY_CAPTURE started delayFromExpectedMs=-", output)

    def test_low_plantnet_score_uses_uncertainty_wording(self):
        result = PlantIdResult(success=True, common_name="Purple coneflower", score=0.05)
        text = plant_response_from_id(result, Config(plantnet_min_confidence=0.25))
        self.assertIn("could not identify it reliably", text)
        self.assertIn("closest low-confidence match was Purple coneflower", text)
        self.assertNotIn("This looks like Purple coneflower", text)

    def test_high_confidence_plant_preserves_confident_wording(self):
        result = PlantIdResult(success=True, common_name="Purple coneflower", score=0.90)
        text = plant_response_from_id(result, Config(plantnet_min_confidence=0.25, plantnet_high_confidence=0.70))
        self.assertIn("This looks like Purple coneflower", text)

    def test_rtsp_disconnect_logs_one_compact_state_transition(self):
        ingest = RtmpAudioIngest("rtmp://example/live/mentra-live")
        self.assertFalse(ingest._reconnecting)
        ingest._reconnecting = True
        ingest._attempt = 1
        self.assertTrue(ingest._reconnecting)

    def test_reconnect_attempts_do_not_overlap(self):
        camera = _OpenCvStreamCamera("rtmp://example/live/mentra-live")
        camera._reconnecting = True
        camera._reconnect_attempt = 1
        camera._miss_count = 15
        camera._next_reconnect_at = 100.0
        with patch("src.camera.time.monotonic", return_value=50.0), patch.object(camera, "_open_capture") as open_capture:
            self.assertIsNone(camera.read())
        open_capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
