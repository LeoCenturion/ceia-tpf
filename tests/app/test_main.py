import unittest
import requests
import requests_mock
from click.testing import CliRunner
from unittest.mock import patch
from src.app.main import cli

class TestMainCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch('os.path.exists', return_value=True)
    @requests_mock.Mocker()
    def test_status_command_running(self, mock_exists, m):
        m.get('http://127.0.0.1:5000/status', json={'status': 'running'})
        result = self.runner.invoke(cli, ['status'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("'status': 'running'", result.output)

    @patch('os.path.exists', return_value=True)
    @requests_mock.Mocker()
    def test_status_command_connection_error(self, mock_exists, m):
        m.get('http://127.0.0.1:5000/status', exc=requests.exceptions.ConnectionError)
        result = self.runner.invoke(cli, ['status'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Could not connect", result.output)

    @patch('os.path.exists', return_value=True)
    @requests_mock.Mocker()
    def test_stop_command(self, mock_exists, m):
        m.post('http://127.0.0.1:5000/shutdown', json={'status': 'shutting down...'})
        with patch('os.remove'): # Mock os.remove to avoid file not found error
             result = self.runner.invoke(cli, ['stop'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Shutdown signal sent", result.output)

if __name__ == '__main__':
    unittest.main()
