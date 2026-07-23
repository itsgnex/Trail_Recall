import unittest
from unittest.mock import patch

import numpy as np

from src.config import Config
from src.intent import Intent
from src.session_state import SessionState
import main


def crop(value):
    return np.full((80, 80, 3), value, dtype=np.uint8)


class VisualPromptStateTests(unittest.TestCase):
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
        self.assertEqual(state.visual_prompt_state, main.VISUAL_COOLDOWN)
        self.assertTrue(listen.call_args.kwargs["after_tts"])

    def test_no_ends_the_interaction(self):
        state = SessionState(visual_prompt_state=main.VISUAL_PROMPT_PLAYING)
        with patch("main.speak") as speak, patch("main.listen", return_value="no"), patch(
            "main.classify_intent_with_source", return_value=(Intent.CANCEL, "test")
        ), patch("main.answer_for") as answer:
            main.handle_trigger("plant", crop(70), Config(), state, camera=None, crop_path="plant.jpg")
        answer.assert_not_called()
        self.assertLessEqual(speak.call_count, 2)
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
        main.visual_mark_absence("other", config, state, now=8.0)
        main.visual_mark_absence("other", config, state, now=12.0)
        self.assertTrue(main.visual_prompt_should_start("plant", image, config, state, now=13.0))


if __name__ == "__main__":
    unittest.main()
