import unittest
from unittest.mock import patch, MagicMock
import requests_mock
from click.testing import CliRunner
from src.app.main import cli

class TestMainCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.p_open_patcher = patch('subprocess.Popen')
        self.mock_p_open = self.p_open_patcher.start()

    def tearDown(self):
        self.p_open_patcher.stop()

import requests
from requests.exceptions import ConnectionError

class TestMainCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.p_open_patcher = patch('subprocess.Popen')
        self.mock_p_open = self.p_open_patcher.start()

    def tearDown(self):
        self.p_open_patcher.stop()

    @requests_mock.Mocker()
    def test_start_command_when_stopped(self, m):
        """Test the start command when the server is not running."""
        m.get('http://127.0.0.1:5000/status', exc=ConnectionError)
        result = self.runner.invoke(cli, ['start'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Bot server started in the background.", result.output)
        self.mock_p_open.assert_called_once()

    @requests_mock.Mocker()
    def test_start_command_when_running(self, m):
        """Test the start command when the server is already running."""
        m.get('http://127.0.0.1:5000/status', json={'status': 'running'})
        result = self.runner.invoke(cli, ['start'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Bot server is already running.", result.output)
        self.mock_p_open.assert_not_called()

    @requests_mock.Mocker()
    def test_stop_command(self, m):
        """Test the stop command."""
        m.post('http://127.0.0.1:5000/shutdown', json={'status': 'shutting down...'})
        result = self.runner.invoke(cli, ['stop'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Shutdown signal sent successfully.", result.output)

    @requests_mock.Mocker()
    def test_status_command_running(self, m):
        """Test the status command when the bot is running."""
        m.get('http://127.0.0.1:5000/status', json={'status': 'running'})
        result = self.runner.invoke(cli, ['status'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("'status': 'running'", result.output)

if __name__ == '__main__':
    unittest.main()
