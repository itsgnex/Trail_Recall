import unittest
from unittest.mock import patch

import numpy as np

from src.config import Config
from src.intent import Intent
from src.llm import describe_current_object
from src.vision_llm import confirm_visual_candidate
import main


class VisualGeminiGateTests(unittest.TestCase):
    def setUp(self):
        self.crop = np.zeros((80, 80, 3), dtype=np.uint8)

    def test_accepts_a_clear_high_confidence_sign(self):
        response = {"image_type": "symbol_sign", "is_clear_enough": True, "confidence": 0.9}
        with patch("src.vision_llm.analyze_image_with_gemma", return_value=response):
            self.assertTrue(confirm_visual_candidate(self.crop, "sign", Config()))

    def test_rejects_an_other_result(self):
        response = {"image_type": "other", "is_clear_enough": True, "confidence": 0.99}
        with patch("src.vision_llm.analyze_image_with_gemma", return_value=response):
            self.assertFalse(confirm_visual_candidate(self.crop, "sign", Config()))

    def test_rejects_a_low_confidence_plant(self):
        response = {"image_type": "plant", "is_clear_enough": True, "confidence": 0.84}
        with patch("src.vision_llm.analyze_image_with_gemma", return_value=response):
            self.assertFalse(confirm_visual_candidate(self.crop, "plant", Config()))

    def test_describes_a_clear_other_object_without_ollama(self):
        response = {"image_type": "other", "description": "I can see a laptop computer on a desk.", "confidence": 0.9}
        with patch("src.llm.analyze_image_with_gemma", return_value=response), patch("src.llm.generate") as generate:
            answer = describe_current_object(self.crop, Config())
        self.assertEqual(answer, "I can see a laptop computer on a desk.")
        generate.assert_not_called()

    def test_generic_object_intents_do_not_inherit_a_sign_label(self):
        self.assertEqual(main._vision_kind_for_intent(Intent.WHAT_AM_I_LOOKING_AT), "object")
        self.assertEqual(main._vision_kind_for_intent(Intent.DESCRIBE_CURRENT_OBJECT), "object")
        self.assertEqual(main._vision_kind_for_intent(Intent.READ_SIGN_TEXT), "sign")
        self.assertEqual(main._vision_kind_for_intent(Intent.IDENTIFY_PLANT), "plant")

    def test_navigation_choice_uses_longer_noise_tolerant_capture(self):
        settings = main._navigation_choice_settings("Would you like me to end the trail or take you back?", Config())
        self.assertEqual(settings, (5.0, 900))


if __name__ == "__main__":
    unittest.main()
