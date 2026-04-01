import unittest
import yaml
import os

class TestConfig(unittest.TestCase):
    def test_load_config(self):
        # Create a dummy config file for testing
        config_data = {
            'bot': {'name': 'Test Bot'},
            'exchange': {'name': 'binance_testnet'}
        }
        with open('config.yaml', 'w') as f:
            yaml.dump(config_data, f)

        # In a real app, the config loading logic would be in a function.
        # We'll simulate that here.
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        self.assertEqual(config['bot']['name'], 'Test Bot')
        self.assertEqual(config['exchange']['name'], 'binance_testnet')

        # Clean up the dummy config file
        os.remove('config.yaml')

if __name__ == '__main__':
    unittest.main()
