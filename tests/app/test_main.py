import unittest
import requests_mock
from click.testing import CliRunner
from src.app.main import cli
import os
from unittest.mock import patch

class TestMainCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch('os.path.exists', return_value=True)
    def test_status_command_running(self, mock_exists):
        with requests_mock.Mocker() as m:
            m.get('http://127.0.0.1:5000/status', json={'status': 'running'})
            result = self.runner.invoke(cli, ['status'])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("'status': 'running'", result.output)
            
    @patch('os.path.exists', return_value=True)
    def test_stop_command(self, mock_exists):
        with requests_mock.Mocker() as m:
            m.post('http://127.0.0.1:5000/shutdown', json={'status': 'shutting down...'})
            with patch('os.remove'):
                result = self.runner.invoke(cli, ['stop'])
                self.assertEqual(result.exit_code, 0)
                self.assertIn("Shutdown signal sent successfully.", result.output)

if __name__ == '__main__':
    unittest.main()
