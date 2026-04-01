import unittest
from click.testing import CliRunner
from src.app.main import cli

class TestMain(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_start_command(self):
        result = self.runner.invoke(cli, ['start'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Starting bot...', result.output)

    def test_stop_command(self):
        result = self.runner.invoke(cli, ['stop'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Stopping bot...', result.output)

    def test_status_command(self):
        result = self.runner.invoke(cli, ['status'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Bot status: running', result.output)

if __name__ == '__main__':
    unittest.main()
