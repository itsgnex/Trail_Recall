import argparse
import io
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from main import resolve_camera_source
from src.config import DEFAULT_ANDROID_TRAIL_URL, DEFAULT_MENTRA_RTSP_URL, Config
from src.intent import Intent, is_trail_command
import src.trail_phone as trail_phone


class BridgeStartupTests(unittest.TestCase):
    def test_default_camera_source_is_mentra(self):
        args = argparse.Namespace(camera=None, camera_index=None, camera_source="mentra")
        source, mode = resolve_camera_source(args)
        self.assertEqual(mode, "mentra")
        self.assertEqual(source, DEFAULT_MENTRA_RTSP_URL)

    def test_local_camera_requires_explicit_index(self):
        args = argparse.Namespace(camera=None, camera_index=None, camera_source="local")
        with self.assertRaises(SystemExit):
            resolve_camera_source(args)

    def test_legacy_camera_is_explicit_local(self):
        args = argparse.Namespace(camera=2, camera_index=None, camera_source="mentra")
        source, mode = resolve_camera_source(args)
        self.assertEqual((source, mode), (2, "local"))

    def test_android_bridge_success_prints_connected(self):
        config = Config(android_trail_base_url=DEFAULT_ANDROID_TRAIL_URL)
        with patch.object(trail_phone, "check_trail_health", return_value=(True, "ok")), patch.object(
            trail_phone, "check_trail_status", return_value=(True, '{"ok":true}')
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                ok, _ = trail_phone.android_bridge_diagnostics(config)
        self.assertTrue(ok)
        self.assertIn("status=CONNECTED", out.getvalue())

    def test_android_bridge_failure_does_not_crash(self):
        config = Config(android_trail_base_url=DEFAULT_ANDROID_TRAIL_URL)
        with patch.object(trail_phone, "check_trail_health", return_value=(False, "timeout")):
            out = io.StringIO()
            with redirect_stdout(out):
                ok, detail = trail_phone.android_bridge_diagnostics(config)
        self.assertFalse(ok)
        self.assertEqual(detail, "timeout")
        self.assertIn("status=UNAVAILABLE", out.getvalue())

    def test_new_navigation_commands_are_recognized(self):
        self.assertEqual(is_trail_command("record my route"), Intent.START_TRAIL)
        self.assertEqual(is_trail_command("stop recording"), Intent.STOP_TRAIL)
        self.assertEqual(is_trail_command("take me back"), Intent.NAVIGATE_BACK)
        self.assertEqual(is_trail_command("choose left"), Intent.CHOOSE_LEFT)
        self.assertEqual(is_trail_command("choose right"), Intent.CHOOSE_RIGHT)
        self.assertEqual(is_trail_command("I reached the destination"), Intent.DESTINATION_REACHED)

    def test_startup_scripts_have_valid_shell_syntax(self):
        for script in ("scripts/run_rtmp_server.sh", "scripts/start_mentra_stack.sh", "scripts/start_android_hotspot.sh"):
            result = subprocess.run(["bash", "-n", script], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
