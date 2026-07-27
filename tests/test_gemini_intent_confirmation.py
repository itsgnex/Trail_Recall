import unittest
from unittest.mock import patch

from src.config import Config
from src.intent import GEMINI_INTENT_CONFIDENCE, Intent, classify_intent_with_source
from src.session_state import SessionState
import main


class GeminiIntentConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(openrouter_api_key="test-key")

    def test_deterministic_intent_does_not_call_gemini(self):
        with patch("src.intent.generate_openrouter") as generate:
            intent, source = classify_intent_with_source("start the trail", self.config)
        self.assertEqual((intent, source), (Intent.START_TRAIL, "trail_rule"))
        generate.assert_not_called()

    def test_gemini_threshold_is_point_eight_nine(self):
        self.assertEqual(GEMINI_INTENT_CONFIDENCE, 0.89)

    def test_high_confidence_start_trail_uses_existing_trail_handler(self):
        with patch("src.intent.generate_openrouter", return_value='{"intent":"START_TRAIL","confidence":0.91}') as generate, patch(
            "main.send_trail_command", return_value=(True, "start")
        ) as send, patch("main.speak"):
            intent, source = classify_intent_with_source("start the", self.config, interaction_context={"interaction_type": "wake_command"})
            handled = main.handle_trail_intent(intent, self.config, SessionState())
        self.assertEqual((intent, source), (Intent.START_TRAIL, "gemini_intent_confirmation"))
        self.assertTrue(handled)
        send.assert_called_once_with("start", self.config)
        self.assertIn("Interaction type: wake_command", generate.call_args.args[0])

    def test_high_confidence_navigate_back_leaves_initial_guidance_to_android(self):
        with patch("src.intent.generate_openrouter", return_value='{"intent":"NAVIGATE_BACK","confidence":0.91}'), patch(
            "main.send_trail_command", return_value=(True, "navigate-back")
        ), patch("main.speak") as speak:
            intent, _ = classify_intent_with_source("take me", self.config)
            handled = main.handle_trail_intent(intent, self.config, SessionState())
        self.assertTrue(handled)
        speak.assert_not_called()

    def test_low_confidence_preserves_clarification(self):
        with patch("src.intent.generate_openrouter", return_value='{"intent":"START_TRAIL","confidence":0.88}'):
            intent, source = classify_intent_with_source("start the", self.config)
        self.assertEqual((intent, source), (Intent.ASK_CLARIFICATION, "fallback"))

    def test_threshold_accepts_point_eight_nine_and_one(self):
        for confidence in (0.89, 1.00):
            with self.subTest(confidence=confidence), patch(
                "src.intent.generate_openrouter", return_value='{"intent":"START_TRAIL","confidence":' + str(confidence) + "}"
            ):
                intent, source = classify_intent_with_source("start the", self.config)
            self.assertEqual((intent, source), (Intent.START_TRAIL, "gemini_intent_confirmation"))

    def test_take_care_does_not_stop_trail_when_gemini_is_below_threshold(self):
        with patch("src.intent.generate_openrouter", return_value='{"intent":"STOP_TRAIL","confidence":0.88}'):
            intent, source = classify_intent_with_source("take care", self.config)
        self.assertEqual((intent, source), (Intent.ASK_CLARIFICATION, "fallback"))

    def test_invalid_json_preserves_clarification(self):
        with patch("src.intent.generate_openrouter", return_value="not json"):
            intent, source = classify_intent_with_source("take me", self.config)
        self.assertEqual((intent, source), (Intent.ASK_CLARIFICATION, "fallback"))

    def test_api_failure_preserves_clarification(self):
        with patch("src.intent.generate_openrouter", return_value=""):
            intent, source = classify_intent_with_source("take me", self.config)
        self.assertEqual((intent, source), (Intent.ASK_CLARIFICATION, "fallback"))

    def test_timeout_preserves_clarification(self):
        with patch("src.intent.generate_openrouter", side_effect=TimeoutError):
            intent, source = classify_intent_with_source("take me", self.config)
        self.assertEqual((intent, source), (Intent.ASK_CLARIFICATION, "fallback"))

    def test_ambiguous_flow_does_not_call_ollama(self):
        with patch("src.intent.generate_openrouter", return_value=""), patch(
            "src.ollama_client.generate_json", side_effect=AssertionError("Ollama must not be used")
        ):
            intent, _ = classify_intent_with_source("start the", self.config)
        self.assertEqual(intent, Intent.ASK_CLARIFICATION)

    def test_sign_and_plant_clarifications_are_context_specific(self):
        sign_state = SessionState(last_detected_kind="sign", visual_prompt_state=main.VISUAL_WAITING_FOR_REPLY)
        plant_state = SessionState(last_detected_kind="plant", visual_prompt_state=main.VISUAL_WAITING_FOR_REPLY)
        self.assertEqual(main._clarification_for_context(sign_state), "Would you like me to read the sign?")
        self.assertEqual(main._clarification_for_context(plant_state), "Would you like me to identify the plant?")


if __name__ == "__main__":
    unittest.main()
