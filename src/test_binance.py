import os
from src.app.exchange import BinanceClient

client = BinanceClient(
    api_key=os.environ.get("BINANCE_TESTNET_API_KEY"),
    api_secret=os.environ.get("BINANCE_TESTNET_API_SECRET"),
    testnet=True
)
print(f"Connection successful: {client.test_connection()}")
print(f"Latest BTCUSDT price: {client.get_latest_price('BTCUSDT')}")
