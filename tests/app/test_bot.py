import unittest
from unittest.mock import MagicMock
from src.app.bot import Bot

class TestBot(unittest.TestCase):
    def test_bot_instantiation(self):
        try:
            bot = Bot(config=None, exchange_client=None, strategy=None)
            self.assertIsInstance(bot, Bot)
        except Exception as e:
            self.fail(f"Bot instantiation raised {e.__class__.__name__} unexpectedly!")

    def test_trading_loop(self):
        # Mock dependencies
        mock_config = {
            'bot': {'symbol': 'BTCUSDT', 'interval': '1m', 'start_str': '1 day ago UTC'}
        }
        mock_exchange_client = MagicMock()
        mock_strategy = MagicMock()

        # Set up mock returns
        mock_exchange_client.get_historical_klines.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        mock_strategy.get_signal.return_value = "BUY"
        mock_strategy.get_order_size.return_value = 0.01

        # Instantiate bot and run loop once
        bot = Bot(config=mock_config, exchange_client=mock_exchange_client, strategy=mock_strategy)
        bot.run()

        # Assert that methods were called
        mock_exchange_client.get_historical_klines.assert_called_once()
        mock_strategy.get_signal.assert_called_once()
        mock_exchange_client.create_order.assert_called_once_with(
            symbol='BTCUSDT', side='BUY', type='MARKET', quantity=0.01
        )

if __name__ == '__main__':
    unittest.main()
