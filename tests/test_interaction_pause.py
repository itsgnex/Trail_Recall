import time
import unittest
from unittest.mock import patch

import numpy as np

from src.audio_http import update_interaction_pause
from src.config import Config
from src.interaction_pause import (
    interaction_accepts_decision_choice,
    interaction_is_paused,
    pause_interaction,
    reset_interaction_pause_for_tests,
)
from src.session_state import SessionState
import main


class InteractionPauseTests(unittest.TestCase):
    def setUp(self):
        reset_interaction_pause_for_tests()

    def tearDown(self):
        reset_interaction_pause_for_tests()

    def test_pause_and_resume_endpoints_update_in_memory_state(self):
        self.assertTrue(update_interaction_pause("/interaction/pause", {"reason": "android_decision_point"})["paused"])
        self.assertTrue(interaction_is_paused())
        self.assertFalse(update_interaction_pause("/interaction/resume", {"reason": "android_decision_point"})["paused"])

    def test_pause_expires_after_timeout(self):
        pause_interaction(timeout_seconds=0.01)
        time.sleep(0.02)
        self.assertFalse(interaction_is_paused())

    def test_wake_command_is_skipped_while_paused(self):
        pause_interaction()
        with patch("main.speak") as speak:
            main.handle_voice_command("hey trail start the trail", Config(), SessionState(), camera=None)
        speak.assert_not_called()

    def test_glasses_saved_route_command_is_allowed_during_android_decision_pause(self):
        pause_interaction("android_decision_point")
        self.assertTrue(interaction_accepts_decision_choice())
        with patch("main.send_trail_command", return_value=(True, "choose-saved-route")) as send:
            main.handle_voice_command("hey trail take saved path", Config(), SessionState(), camera=None)
        send.assert_called_once_with("choose-saved-route", unittest.mock.ANY)

    def test_bare_wake_at_decision_point_answers_then_accepts_saved_route(self):
        pause_interaction("android_decision_point")
        with patch("main.listen", return_value="take saved path"), patch("main.speak") as speak, patch(
            "main.send_trail_command", return_value=(True, "choose-saved-route")
        ) as send:
            main.handle_voice_command("hey trail", Config(), SessionState(), camera=None)
        speak.assert_called_once_with("Yes?")
        send.assert_called_once_with("choose-saved-route", unittest.mock.ANY)

    def test_plant_and_sign_prompts_are_skipped_while_paused(self):
        pause_interaction()
        crop = np.zeros((40, 40, 3), dtype=np.uint8)
        self.assertFalse(main.visual_prompt_should_start("plant", crop, Config(), SessionState()))
        self.assertFalse(main.visual_prompt_should_start("sign", crop, Config(), SessionState()))

    def test_pause_does_not_stop_stream_ingest(self):
        from src.rtmp_audio_ingest import RtmpAudioIngest

        ingest = RtmpAudioIngest("rtmp://127.0.0.1:1935/live/mentra-live")
        pause_interaction()
        self.assertTrue(interaction_is_paused())
        self.assertFalse(ingest._stop.is_set())


if __name__ == "__main__":
    unittest.main()
