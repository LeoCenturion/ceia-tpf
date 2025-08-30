import os
import time
import pandas as pd
from binance.client import Client
from binance.enums import *
import csv
from datetime import datetime

# API credentials from environment variables
# Make sure to set BINANCE_API_KEY and BINANCE_API_SECRET in your environment
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')

if not API_KEY or not API_SECRET:
    raise ValueError("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")

# Initialize Binance client for testnet
client = Client(API_KEY, API_SECRET, testnet=True)

TRANSACTION_LOG_FILE = 'transactions.csv'

# Trading parameters
symbol = 'BTCUSDT'
timeframe = '1m'  # Using 1 minute timeframe for historical data


def log_transaction(timestamp, side, price, quantity, value):
    """Logs a transaction to the CSV file."""
    file_exists = os.path.isfile(TRANSACTION_LOG_FILE)
    with open(TRANSACTION_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'side', 'price', 'quantity', 'value'])
        
        # Convert ms timestamp to readable format
        ts = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([ts, side, price, quantity, value])


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


def execute_trade(signal, symbol, trade_amount, open_position_quantity, current_price, max_capital):
    """
    Executes a trade based on the signal.
    Returns the quantity of the open position.
    """
    current_position_value = open_position_quantity * current_price

    if signal == 'BUY':
        if current_position_value + trade_amount <= max_capital:
            print(f"Executing BUY order for {trade_amount} USDT worth of {symbol}")
            try:
                # For SPOT market orders, we can specify quoteOrderQty for the amount in quote currency (USDT)
                order = client.create_order(
                    symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quoteOrderQty=trade_amount)
                print(order)

                # Log the transaction
                executed_qty = float(order['executedQty'])
                cummulative_quote_qty = float(order['cummulativeQuoteQty'])
                avg_price = cummulative_quote_qty / executed_qty
                log_transaction(order['transactTime'], 'BUY', avg_price, executed_qty, cummulative_quote_qty)

                return open_position_quantity + executed_qty  # Return the new total quantity
            except Exception as e:
                print(f"An error occurred during BUY: {e}")
                return open_position_quantity  # No change in position
        else:
            print(f"BUY signal ignored. Order would exceed max capital. Current value: {current_position_value:.2f} USDT, Max capital: {max_capital} USDT.")
            return open_position_quantity
    elif signal == 'SELL' and open_position_quantity > 0:
        print(f"Executing SELL order for {open_position_quantity} {symbol}")
        try:
            # When selling, we specify the quantity of the base asset (BTC)
            order = client.create_order(
                symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=open_position_quantity)
            print(order)

            # Log the transaction
            executed_qty = float(order['executedQty'])
            cummulative_quote_qty = float(order['cummulativeQuoteQty'])
            avg_price = cummulative_quote_qty / executed_qty if executed_qty > 0 else 0
            log_transaction(order['transactTime'], 'SELL', avg_price, executed_qty, cummulative_quote_qty)

            return 0.0  # Position is now closed
        except Exception as e:
            print(f"An error occurred during SELL: {e}")
            return open_position_quantity  # Position is still open
    return open_position_quantity  # No trade executed, return current position status


def main(strategy_func, trade_amount, max_capital):
    """
    Main trading loop.
    """
    open_position_quantity = 0.0
    print("Starting trading bot...")
    print(f"Using strategy: {strategy_func.__name__}")
    print(f"Trade amount: {trade_amount} USDT")
    print(f"Max capital: {max_capital} USDT")

    try:
        while True:
            try:
                # We need enough data for MACD calculation, e.g., long_window + signal_window
                df = get_historical_data(symbol, timeframe, "100 minutes ago UTC")
                if len(df) < 26 + 9:  # not enough data for MACD
                    print("Not enough historical data yet. Waiting...")
                    time.sleep(60)
                    continue

                signal = strategy_func(df)
                current_price = df['close'].iloc[-1]
                current_position_value = open_position_quantity * current_price
                position_open_str = f"Yes, size: {open_position_quantity:.6f} BTC ({current_position_value:.2f} USDT)" if open_position_quantity > 0 else "No"
                print(f"Timestamp: {pd.Timestamp.now()}, Price: {current_price:.2f}, Signal: {signal}, Position: {position_open_str}")

                open_position_quantity = execute_trade(signal, symbol, trade_amount, open_position_quantity, current_price, max_capital)

                # Wait for the next candle
                time.sleep(60)
            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                # Wait a bit before retrying to avoid spamming on persistent errors
                time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopping bot...")
    finally:
        if open_position_quantity > 0:
            print(f"Closing open position of {open_position_quantity} {symbol}...")
            try:
                order = client.create_order(
                    symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=open_position_quantity)
                print("Position closed successfully.")
                print(order)

                # Log the final transaction
                executed_qty = float(order['executedQty'])
                cummulative_quote_qty = float(order['cummulativeQuoteQty'])
                avg_price = cummulative_quote_qty / executed_qty if executed_qty > 0 else 0
                log_transaction(order['transactTime'], 'SELL', avg_price, executed_qty, cummulative_quote_qty)
            except Exception as e:
                print(f"An error occurred while closing position on exit: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='A simple Binance trading bot.')
    parser.add_argument('--trade-amount', type=float, default=15.0,
                        help='The amount in USDT to use for each trade. Default is 15 USDT. '
                             'Note: Binance has minimum order sizes.')
    parser.add_argument('--max-capital', type=float, default=1000.0,
                        help='The maximum available capital in USDT to use for positions. Default is 1000 USDT.')
    args = parser.parse_args()

    if args.trade_amount < 11:
        # Binance minimum order size is often around 10 USDT. Setting a bit higher.
        print("Warning: trade-amount is very low. Binance might reject the order. Recommended: > 11 USDT.")

    if args.trade_amount > args.max_capital:
        raise ValueError("trade-amount cannot be greater than max-capital.")

    # You can define other strategies and switch them here
    selected_strategy = macd_strategy
    main(selected_strategy, args.trade_amount, args.max_capital)
