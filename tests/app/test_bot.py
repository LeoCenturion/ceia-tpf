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
            'bot': {'symbol': 'BTCUSDT', 'interval': '1m', 'start_str': '1 day ago UTC'},
            'risk_management': {'stop_loss': 0.1, 'max_drawdown': 0.2},
            'capital_allocation': {'max_capital': 1000}
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
    
    def test_risk_management_stop_loss(self):
        # Mock dependencies
        mock_config = {
            'bot': {'symbol': 'BTCUSDT', 'interval': '1m', 'start_str': '1 day ago UTC'},
            'risk_management': {'stop_loss': 0.1, 'max_drawdown': 0.2},
            'capital_allocation': {'max_capital': 1000}
        }
        mock_exchange_client = MagicMock()
        mock_strategy = MagicMock()

        # Set up mock returns
        mock_exchange_client.get_historical_klines.return_value = [[1, 2, 3, 4, 100, 6, 7, 8, 9, 10, 11, 12]]
        mock_strategy.get_signal.return_value = "SELL" # Should trigger stop-loss
        mock_strategy.get_order_size.return_value = 0.01

        # Instantiate bot and run loop once
        bot = Bot(config=mock_config, exchange_client=mock_exchange_client, strategy=mock_strategy)
        bot.account_balance = 1000
        bot.position = {'price': 110, 'quantity': 1} # Losing position
        bot.run()
        
        # Assert that a SELL order was created to close the position
        mock_exchange_client.create_order.assert_called_once_with(
            symbol='BTCUSDT', side='SELL', type='MARKET', quantity=1
        )

    def test_capital_allocation(self):
        # Mock dependencies
        mock_config = {
            'bot': {'symbol': 'BTCUSDT', 'interval': '1m', 'start_str': '1 day ago UTC'},
            'risk_management': {'stop_loss': 0.1, 'max_drawdown': 0.2},
            'capital_allocation': {'max_capital': 100} # Low capital to trigger the check
        }
        mock_exchange_client = MagicMock()
        mock_strategy = MagicMock()

        # Set up mock returns
        mock_exchange_client.get_historical_klines.return_value = [[1, 2, 3, 4, 50, 6, 7, 8, 9, 10, 11, 12]]
        mock_strategy.get_signal.return_value = "BUY"
        mock_strategy.get_order_size.return_value = 3 # This would exceed max_capital

        # Instantiate bot and run loop once
        bot = Bot(config=mock_config, exchange_client=mock_exchange_client, strategy=mock_strategy)
        bot.run()
        
        # Assert that create_order was not called
        mock_exchange_client.create_order.assert_not_called()

if __name__ == '__main__':
    unittest.main()
