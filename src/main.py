import os
import time
import pandas as pd
from binance.client import Client
from binance.enums import *

# API credentials from environment variables
# Make sure to set BINANCE_API_KEY and BINANCE_API_SECRET in your environment
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')

if not API_KEY or not API_SECRET:
    raise ValueError("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")

# Initialize Binance client for testnet
client = Client(API_KEY, API_SECRET, testnet=True)

# Trading parameters
symbol = 'BTCUSDT'
quantity = 0.001  # Amount of BTC to buy/sell
timeframe = '1m'  # Using 1 minute timeframe for historical data


def get_historical_data(symbol, interval, lookback):
    """
    Fetches historical klines from Binance.
    """
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['close'] = pd.to_numeric(df['close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def macd_strategy(df: pd.DataFrame, short_window=12, long_window=26, signal_window=9):
    """
    MACD trading strategy.
    Returns 'BUY', 'SELL' or 'HOLD'.
    """
    df['short_ema'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['long_ema'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['macd'] = df['short_ema'] - df['long_ema']
    df['signal_line'] = df['macd'].ewm(span=signal_window, adjust=False).mean()

    # Check for crossover
    if df['macd'].iloc[-1] > df['signal_line'].iloc[-1] and df['macd'].iloc[-2] <= df['signal_line'].iloc[-2]:
        return 'BUY'
    elif df['macd'].iloc[-1] < df['signal_line'].iloc[-1] and df['macd'].iloc[-2] >= df['signal_line'].iloc[-2]:
        return 'SELL'
    else:
        return 'HOLD'


def execute_trade(signal, symbol, quantity, position_open):
    """
    Executes a trade based on the signal.
    """
    if signal == 'BUY' and not position_open:
        print(f"Executing BUY order for {quantity} {symbol}")
        try:
            # Use create_test_order for testing on testnet without executing
            # order = client.create_test_order(
            #     symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=quantity)
            order = client.create_order(
                symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=quantity)
            print(order)
            return True  # Position is now open
        except Exception as e:
            print(f"An error occurred during BUY: {e}")
            return False  # Position is not open
    elif signal == 'SELL' and position_open:
        print(f"Executing SELL order for {quantity} {symbol}")
        try:
            # Use create_test_order for testing on testnet without executing
            # order = client.create_test_order(
            #     symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=quantity)
            order = client.create_order(
                symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=quantity)
            print(order)
            return False  # Position is now closed
        except Exception as e:
            print(f"An error occurred during SELL: {e}")
            return True  # Position is still open
    return position_open  # No trade executed, return current position status


def main(strategy_func):
    """
    Main trading loop.
    """
    position_open = False
    print("Starting trading bot...")
    print(f"Using strategy: {strategy_func.__name__}")

    while True:
        try:
            # We need enough data for MACD calculation, e.g., long_window + signal_window
            df = get_historical_data(symbol, timeframe, "100 minutes ago UTC")
            if len(df) < 26 + 9:  # not enough data for MACD
                print("Not enough historical data yet. Waiting...")
                time.sleep(60)
                continue

            signal = strategy_func(df)
            print(f"Timestamp: {pd.Timestamp.now()}, Signal: {signal}, Position Open: {position_open}")

            position_open = execute_trade(signal, symbol, quantity, position_open)

            # Wait for the next candle
            time.sleep(60)
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            # Wait a bit before retrying to avoid spamming on persistent errors
            time.sleep(60)


if __name__ == '__main__':
    # You can define other strategies and switch them here
    selected_strategy = macd_strategy
    main(selected_strategy)
