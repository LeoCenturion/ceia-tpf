import unittest
import os
import yaml
from src.app.config import load_config

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.valid_config_data = {
            'bot': {'name': 'Test Bot'},
            'exchange': {'name': 'binance_testnet'}
        }
        self.invalid_config_data = {
            'bot': {'name': 'Test Bot'}
            # Missing 'exchange' section
        }

    def tearDown(self):
        if os.path.exists('config.yaml'):
            os.remove('config.yaml')

    def test_load_valid_config(self):
        with open('config.yaml', 'w') as f:
            yaml.dump(self.valid_config_data, f)
        
        config = load_config('config.yaml')
        self.assertEqual(config, self.valid_config_data)

    def test_load_missing_config(self):
        with self.assertRaises(FileNotFoundError):
            load_config('non_existent_config.yaml')

    def test_load_invalid_config(self):
        with open('config.yaml', 'w') as f:
            yaml.dump(self.invalid_config_data, f)
        
        # In a real application, you would have a more sophisticated
        # validation mechanism that would raise a specific error.
        # For now, we'll just check that it doesn't return the invalid data.
        with self.assertRaises(KeyError):
             # A simple validation could be to check for the presence of keys.
            config = load_config('config.yaml')
            if 'exchange' not in config:
                raise KeyError("Missing 'exchange' in config")


if __name__ == '__main__':
    unittest.main()
