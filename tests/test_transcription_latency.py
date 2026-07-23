import unittest
from unittest.mock import Mock, patch

from src.config import Config
import src.speech_in as speech_in


class FakeAudio:
    def get_wav_data(self, convert_rate=None, convert_width=None):
        self.convert_rate = convert_rate
        self.convert_width = convert_width
        return b"RIFF....WAVEfmt "


class FakeResponse:
    def __init__(self, status_code=200, text="", json_data=None, json_error=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise self._json_error
        return self._json_data


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


class TranscriptionLatencyTests(unittest.TestCase):
    def test_short_reply_yes(self):
        self.assertTrue(speech_in._is_complete_short_reply("yes"))
        self.assertTrue(speech_in._is_complete_short_reply("Yeah."))

    def test_short_reply_no(self):
        self.assertTrue(speech_in._is_complete_short_reply("no"))
        self.assertTrue(speech_in._is_complete_short_reply("No!"))

    def test_command_and_follow_up_timing_defaults(self):
        config = Config()
        self.assertEqual(config.command_record_seconds, 2.5)
        self.assertEqual(config.follow_up_timeout_seconds, 1.5)
        self.assertEqual(config.follow_up_silence_ms, 300)
        self.assertEqual(config.confirmation_record_seconds, 2.0)
        self.assertEqual(config.confirmation_silence_ms, 350)
        self.assertEqual(config.speech_end_silence_ms, 450)

    @patch("requests.post", return_value=FakeResponse(json_data={"text": "yes"}))
    def test_openrouter_json_success(self, post):
        audio = FakeAudio()
        config = Config(openrouter_api_key="present")
        text = speech_in.transcribe_with_openrouter(audio, config)
        self.assertEqual(text, "yes")
        self.assertEqual(audio.convert_rate, 16000)
        self.assertEqual(audio.convert_width, 2)
        self.assertEqual(post.call_args.kwargs["data"]["response_format"], "json")
        self.assertEqual(post.call_args.kwargs["timeout"], 3.0)

    @patch("src.speech_in.transcribe_with_whisper", return_value="local fallback")
    @patch("requests.post", return_value=FakeResponse(json_data={"text": "yes"}))
    def test_openrouter_json_success_does_not_fallback(self, _post, whisper):
        config = Config(openrouter_api_key="present")
        text = speech_in._transcribe_primary(FakeAudio(), config)
        self.assertEqual(text, "yes")
        whisper.assert_not_called()

    @patch("requests.post", return_value=FakeResponse(json_data={"duration": 0.4, "segments": [{"text": "no"}]}))
    def test_openrouter_verbose_json_success(self, _post):
        config = Config(openrouter_api_key="present")
        text = speech_in.transcribe_with_openrouter(FakeAudio(), config)
        self.assertEqual(text, "no")

    @patch("src.speech_in.transcribe_with_whisper", return_value="local fallback")
    @patch("requests.post", return_value=FakeResponse(text="{bad json", json_error=ValueError("malformed")))
    def test_malformed_json_uses_local_fallback(self, _post, whisper):
        config = Config(openrouter_api_key="present")
        text = speech_in._transcribe_primary(FakeAudio(), config)
        self.assertEqual(text, "local fallback")
        whisper.assert_called_once()

    @patch("src.speech_in.transcribe_with_whisper", return_value="local fallback")
    @patch("requests.post", return_value=FakeResponse(status_code=400, json_data={"error": "bad format"}))
    def test_http_400_uses_local_fallback(self, _post, whisper):
        config = Config(openrouter_api_key="present")
        text = speech_in._transcribe_primary(FakeAudio(), config)
        self.assertEqual(text, "local fallback")
        whisper.assert_called_once()

    @patch("src.speech_in.transcribe_with_whisper", return_value="local fallback")
    @patch("requests.post", side_effect=TimeoutError("too slow"))
    def test_openrouter_timeout_uses_local_fallback(self, _post, whisper):
        config = Config(openrouter_api_key="present")
        text = speech_in._transcribe_primary(FakeAudio(), config)
        self.assertEqual(text, "local fallback")
        whisper.assert_called_once()

    @patch("src.speech_in.transcribe_with_whisper", return_value="local fallback")
    def test_missing_api_key_uses_local_fallback(self, whisper):
        config = Config(openrouter_api_key="")
        text = speech_in._transcribe_primary(FakeAudio(), config)
        self.assertEqual(text, "local fallback")
        whisper.assert_called_once()

    @patch("src.speech_in.transcribe_with_whisper", return_value="local fallback")
    @patch("requests.post", return_value=Mock(status_code=200, text="   ", json=Mock(return_value={})))
    def test_empty_api_response_uses_local_fallback(self, _post, whisper):
        config = Config(openrouter_api_key="present")
        text = speech_in._transcribe_primary(FakeAudio(), config)
        self.assertEqual(text, "local fallback")
        whisper.assert_called_once()

    def test_yes_audio_is_not_rejected_as_silence(self):
        chunks = [pcm_chunk(100) for _ in range(4)] + [pcm_chunk(0) for _ in range(8)]
        with patch("src.mic_ingest.glasses_mic_buffer", return_value=FakeGlassesBuffer(chunks)):
            audio = speech_in._record_glasses_audio(Config(use_glasses_mic=True), 1.5, quiet=True, silence_ms=300)
        self.assertFalse(isinstance(audio, str))

    def test_no_audio_is_not_rejected_as_silence(self):
        chunks = [pcm_chunk(-100) for _ in range(4)] + [pcm_chunk(0) for _ in range(8)]
        with patch("src.mic_ingest.glasses_mic_buffer", return_value=FakeGlassesBuffer(chunks)):
            audio = speech_in._record_glasses_audio(Config(use_glasses_mic=True), 1.5, quiet=True, silence_ms=300)
        self.assertFalse(isinstance(audio, str))


if __name__ == "__main__":
    unittest.main()
