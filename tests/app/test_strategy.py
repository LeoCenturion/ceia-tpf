import unittest
import pandas as pd
from src.app.strategy import Strategy, MACDStrategy

class TestStrategy(unittest.TestCase):
    def test_strategy_interface(self):
        class DummyStrategy(Strategy):
            def get_signal(self, data):
                return "HOLD"
            def get_order_size(self):
                return 0
        strategy = DummyStrategy()
        self.assertEqual(strategy.get_signal(None), "HOLD")
        self.assertEqual(strategy.get_order_size(), 0)

class TestMACDStrategy(unittest.TestCase):
    def setUp(self):
        # Create sample data
        data = {
            'close': [10, 12, 15, 14, 13, 16, 18, 20, 19, 22, 25, 23]
        }
        self.df = pd.DataFrame(data)
        self.strategy = MACDStrategy(fast_period=3, slow_period=6, signal_period=4)

    def test_macd_calculation(self):
        self.strategy.get_signal(self.df)
        self.assertIn('macd', self.df.columns)
        self.assertIn('macds', self.df.columns)
        self.assertIn('macdh', self.df.columns)

    def test_get_signal(self):
        # This is a simplified test. In a real scenario, you would have
        # more robust data and assertions.
        signal = self.strategy.get_signal(self.df)
        self.assertIn(signal, ["BUY", "SELL", "HOLD"])

if __name__ == '__main__':
    unittest.main()
