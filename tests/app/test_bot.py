import unittest
from src.app.bot import Bot

class TestBot(unittest.TestCase):
    def test_bot_instantiation(self):
        try:
            bot = Bot()
            self.assertIsInstance(bot, Bot)
        except Exception as e:
            self.fail(f"Bot instantiation raised {e.__class__.__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()
