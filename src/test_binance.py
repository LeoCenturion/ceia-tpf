import os
from src.app.exchange import BinanceClient
import pandas as pd
from src.app.strategy import MACDStrategy

client = BinanceClient(
    api_key=os.environ.get("BINANCE_TESTNET_API_KEY"),
    api_secret=os.environ.get("BINANCE_TESTNET_API_SECRET"),
    testnet=True
)
print(f"Connection successful: {client.test_connection()}")
print(f"Latest BTCUSDT price: {client.get_latest_price('BTCUSDT')}")


data = {
    'close': [10, 12, 15, 14, 13, 16, 18, 20, 19, 22, 25, 23, 21, 20, 19, 18, 17, 16]
}
df = pd.DataFrame(data)
strategy = MACDStrategy(fast_period=3, slow_period=6, signal_period=4)
signal = strategy.get_signal(df)
print(f"Generated signal: {signal}")
