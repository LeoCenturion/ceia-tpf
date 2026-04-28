import os
import sys

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.app.bot import Bot
from src.app.config import load_config
from src.app.exchange import BinanceClient
from src.app.strategy import MACDStrategy


def run_manual_test():
    # Ensure environment variables are set for Binance API keys
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET")

    if not api_key or not api_secret:
        print(
            "Please set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET environment variables."
        )
        return

    config = load_config("config.yaml")
    exchange_client = BinanceClient(
        api_key=api_key, api_secret=api_secret, testnet=True
    )
    strategy = MACDStrategy()
    bot = Bot(config, exchange_client, strategy)

    print("Starting manual bot run. Observe console output for actions.")
    print("Note: This will run once and then exit in this manual test setup.")
    # Modify the bot's run method to exit after one loop for manual testing
    bot.run_once = True  # Assuming a modification in Bot.run() for this
    bot.run()


if __name__ == "__main__":
    run_manual_test()
