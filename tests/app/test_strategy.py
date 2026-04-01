import unittest
from src.app.strategy import Strategy

class DummyStrategy(Strategy):
    def get_signal(self):
        return "HOLD"

    def get_order_size(self):
        return 0

class TestStrategy(unittest.TestCase):
    def test_strategy_interface(self):
        strategy = DummyStrategy()
        self.assertEqual(strategy.get_signal(), "HOLD")
        self.assertEqual(strategy.get_order_size(), 0)

if __name__ == '__main__':
    unittest.main()
