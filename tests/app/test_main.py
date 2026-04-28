import os
import tempfile
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from src.app.main import cli


class TestMainCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.p_open_patcher = patch("subprocess.Popen")
        self.mock_p_open = self.p_open_patcher.start()

    def tearDown(self):
        self.p_open_patcher.stop()

    def test_start_command_when_stopped(self):
        """Test the start command when no PID file exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = os.path.join(tmp_dir, "bot.pid")
            with patch("src.app.main.PID_FILE", pid_file):
                result = self.runner.invoke(cli, ["start"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Bot server started in the background", result.output)
        self.mock_p_open.assert_called_once()

    def test_start_command_when_running(self):
        """Test the start command when a PID file already exists."""
        with tempfile.NamedTemporaryFile(suffix=".pid", delete=False) as f:
            pid_file = f.name
        try:
            with patch("src.app.main.PID_FILE", pid_file):
                result = self.runner.invoke(cli, ["start"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn(
                "Bot server is already running (PID file exists).", result.output
            )
            self.mock_p_open.assert_not_called()
        finally:
            if os.path.exists(pid_file):
                os.remove(pid_file)

    def test_stop_command(self):
        """Test the stop command sends SIGTERM and removes PID file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False) as f:
            f.write("12345")
            pid_file = f.name
        try:
            with (
                patch("src.app.main.PID_FILE", pid_file),
                patch("src.app.main.os.kill") as mock_kill,
            ):
                result = self.runner.invoke(cli, ["stop"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Sent shutdown signal to process 12345.", result.output)
            mock_kill.assert_called_once_with(12345, 15)
        finally:
            if os.path.exists(pid_file):
                os.remove(pid_file)

    def test_status_command_running(self):
        """Test the status command when the bot is running."""
        import requests_mock as req_mock

        with (
            req_mock.Mocker() as m,
            patch("src.app.main.os.path.exists", return_value=True),
        ):
            m.get("http://127.0.0.1:5000/status", json={"status": "running"})
            result = self.runner.invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("'status': 'running'", result.output)


if __name__ == "__main__":
    unittest.main()
