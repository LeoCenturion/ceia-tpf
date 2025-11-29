import os
import time
import pandas as pd
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL
import csv
from datetime import datetime
import random
from decimal import Decimal, ROUND_DOWN

# API credentials from environment variables
# Make sure to set BINANCE_API_KEY and BINANCE_API_SECRET in your environment
API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError(
        "Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables."
    )

# Initialize Binance client for testnet
client = Client(API_KEY, API_SECRET, testnet=True)

# Proactively sync time with Binance server to prevent signature errors
try:
    server_time = client.get_server_time()
    client.timestamp_offset = server_time["serverTime"] - int(time.time() * 1000)
    print(f"Time offset with Binance server: {client.timestamp_offset}ms")
except Exception as e:
    print(f"Could not sync time with Binance: {e}")


TRANSACTION_LOG_FILE = "transactions.csv"

# Trading parameters
symbol = "BTCUSDT"
timeframe = "1m"  # Using 1 minute timeframe for historical data


def log_transaction(timestamp, side, price, quantity, value, strategy_name):
    """Logs a transaction to the CSV file."""
    file_exists = os.path.isfile(TRANSACTION_LOG_FILE)
    with open(TRANSACTION_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["timestamp", "side", "price", "quantity", "value", "strategy"]
            )

        # Convert ms timestamp to readable format
        ts = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([ts, side, price, quantity, value, strategy_name])


def get_symbol_info(symbol):
    """
    Fetches the quantity precision (stepSize), minQty, and minNotional for a given symbol.
    """
    info = client.get_exchange_info()
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            min_qty = 0.0
            step_size = 0.0
            min_notional = 0.0
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    min_qty = float(f["minQty"])
                    step_size = float(f["stepSize"])
                elif f["filterType"] == "MIN_NOTIONAL":
                    min_notional = float(f["minNotional"])

            # Calculate decimal places from step_size
            # Example: step_size = 0.001 -> decimal_places = 3
            # Example: step_size = 1.0   -> decimal_places = 0
            decimal_places = 0
            if step_size > 0:
                decimal_places = abs(Decimal(str(step_size)).as_tuple().exponent)

            return {
                "min_qty": min_qty,
                "step_size": step_size,
                "decimal_places": decimal_places,
                "min_notional": min_notional,
            }
    raise ValueError(f"Could not find symbol information for {symbol}")


def get_historical_data(symbol, interval, lookback):
    """
    Fetches historical klines from Binance.
    """
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(
        klines,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df["close"] = pd.to_numeric(df["close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def macd_strategy(df: pd.DataFrame, short_window=12, long_window=26, signal_window=9):
    """
    MACD trading strategy.
    Returns 'BUY', 'SELL' or 'HOLD'.
    """
    df["short_ema"] = df["close"].ewm(span=short_window, adjust=False).mean()
    df["long_ema"] = df["close"].ewm(span=long_window, adjust=False).mean()
    df["macd"] = df["short_ema"] - df["long_ema"]
    df["signal_line"] = df["macd"].ewm(span=signal_window, adjust=False).mean()

    # Check for crossover
    if (
        df["macd"].iloc[-1] > df["signal_line"].iloc[-1]
        and df["macd"].iloc[-2] <= df["signal_line"].iloc[-2]
    ):
        return "BUY"
    elif (
        df["macd"].iloc[-1] < df["signal_line"].iloc[-1]
        and df["macd"].iloc[-2] >= df["signal_line"].iloc[-2]
    ):
        return "SELL"
    else:
        return "HOLD"


def random_strategy(df: pd.DataFrame):
    """
    A random trading strategy.
    Returns 'BUY' or 'SELL' with 50/50 probability.
    """
    if random.random() < 0.5:
        return "BUY"
    else:
        return "SELL"


def execute_trade(
    signal,
    symbol,
    trade_amount,
    open_position_quantity,
    current_price,
    max_capital,
    strategy_name,
):
    """
    Executes a trade based on the signal.
    Returns the quantity of the open position.
    """
    current_position_value = open_position_quantity * current_price

    if signal == "BUY":
        if current_position_value + trade_amount <= max_capital:
            print(f"Executing BUY order for {trade_amount} USDT worth of {symbol}")
            try:
                # For SPOT market orders, we can specify quoteOrderQty for the amount in quote currency (USDT)
                order = client.create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quoteOrderQty=trade_amount,
                )
                print(order)

                # Log the transaction
                executed_qty = float(order["executedQty"])
                cummulative_quote_qty = float(order["cummulativeQuoteQty"])
                avg_price = cummulative_quote_qty / executed_qty
                log_transaction(
                    order["transactTime"],
                    "BUY",
                    avg_price,
                    executed_qty,
                    cummulative_quote_qty,
                    strategy_name,
                )

                return (
                    open_position_quantity + executed_qty
                )  # Return the new total quantity
            except Exception as e:
                print(f"An error occurred during BUY: {e}")
                return open_position_quantity  # No change in position
        else:
            print(
                f"BUY signal ignored. Order would exceed max capital. Current value: {current_position_value:.2f} USDT, Max capital: {max_capital} USDT."
            )
            return open_position_quantity
    elif signal == "SELL" and open_position_quantity > 0:
        # Get symbol info for precision and minimums
        try:
            symbol_info = get_symbol_info(symbol)
            decimal_places = symbol_info["decimal_places"]
            min_qty = symbol_info["min_qty"]
            min_notional = symbol_info["min_notional"]
        except ValueError as e:
            print(f"Error getting symbol info: {e}. Cannot execute SELL order safely.")
            return open_position_quantity

        # Round quantity to the correct precision, rounding down
        open_position_decimal = Decimal(str(open_position_quantity))
        rounding_precision = Decimal("1e-{}".format(decimal_places))
        rounded_quantity = float(
            open_position_decimal.quantize(rounding_precision, rounding=ROUND_DOWN)
        )

        # Check if rounded quantity is valid
        if rounded_quantity <= 0:
            print(
                f"SELL signal ignored. Rounded quantity is {rounded_quantity}, which is too small to trade."
            )
            return open_position_quantity
        if rounded_quantity < min_qty:
            print(
                f"SELL signal ignored. Rounded quantity {rounded_quantity} is less than minimum quantity {min_qty}."
            )
            return open_position_quantity
        if rounded_quantity * current_price < min_notional:
            print(
                f"SELL signal ignored. Notional value {rounded_quantity * current_price:.2f} is less than minimum notional {min_notional}."
            )
            return open_position_quantity

        print(f"Executing SELL order for {rounded_quantity} {symbol}")
        try:
            # When selling, we specify the quantity of the base asset (BTC)
            order = client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=rounded_quantity,
            )
            print(order)

            # Log the transaction
            executed_qty = float(order["executedQty"])
            cummulative_quote_qty = float(order["cummulativeQuoteQty"])
            avg_price = cummulative_quote_qty / executed_qty if executed_qty > 0 else 0
            log_transaction(
                order["transactTime"],
                "SELL",
                avg_price,
                executed_qty,
                cummulative_quote_qty,
                strategy_name,
            )

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
                current_price = df["close"].iloc[-1]
                current_position_value = open_position_quantity * current_price
                position_open_str = (
                    f"Yes, size: {open_position_quantity:.6f} BTC ({current_position_value:.2f} USDT)"
                    if open_position_quantity > 0
                    else "No"
                )
                print(
                    f"Timestamp: {pd.Timestamp.now()}, Price: {current_price:.2f}, Signal: {signal}, Position: {position_open_str}"
                )

                open_position_quantity = execute_trade(
                    signal,
                    symbol,
                    trade_amount,
                    open_position_quantity,
                    current_price,
                    max_capital,
                    strategy_func.__name__,
                )

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
            # Get symbol info for graceful shutdown
            try:
                symbol_info = get_symbol_info(symbol)
                decimal_places = symbol_info["decimal_places"]
                min_qty = symbol_info["min_qty"]
                min_notional = symbol_info["min_notional"]
            except ValueError as e:
                print(
                    f"Error getting symbol info on shutdown: {e}. Attempting to sell with original quantity (may fail)."
                )
                decimal_places = 8  # Fallback to a common high precision
                min_qty = 0.0
                min_notional = 0.0

            open_position_decimal = Decimal(str(open_position_quantity))
            rounding_precision = Decimal("1e-{}".format(decimal_places))
            rounded_quantity = float(
                open_position_decimal.quantize(rounding_precision, rounding=ROUND_DOWN)
            )

            if rounded_quantity <= 0:
                print(
                    f"Warning: Open position quantity {open_position_quantity} rounded to {rounded_quantity} is too small to sell on exit. Position not closed."
                )
                return  # Exit finally without trying to sell 0 quantity
            if rounded_quantity < min_qty:
                print(
                    f"Warning: Open position quantity {rounded_quantity} is less than min_qty {min_qty}. Position not closed."
                )
                return

            # Current price is needed to check min_notional, get it again or assume a reasonable price
            # For graceful shutdown, let's fetch the latest price
            current_price_for_shutdown = 0.0
            try:
                df_latest = get_historical_data(symbol, timeframe, "1 minute ago UTC")
                if not df_latest.empty:
                    current_price_for_shutdown = df_latest["close"].iloc[-1]
            except Exception as e:
                print(
                    f"Warning: Could not get current price for min_notional check on shutdown: {e}. Proceeding without check."
                )

            if (
                current_price_for_shutdown > 0
                and rounded_quantity * current_price_for_shutdown < min_notional
            ):
                print(
                    f"Warning: Notional value {rounded_quantity * current_price_for_shutdown:.2f} is less than minimum notional {min_notional}. Position not closed on exit."
                )
                return

            print(f"Closing open position of {rounded_quantity} {symbol}...")
            try:
                order = client.create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=rounded_quantity,
                )
                print("Position closed successfully.")
                print(order)

                # Log the final transaction
                executed_qty = float(order["executedQty"])
                cummulative_quote_qty = float(order["cummulativeQuoteQty"])
                avg_price = (
                    cummulative_quote_qty / executed_qty if executed_qty > 0 else 0
                )
                log_transaction(
                    order["transactTime"],
                    "SELL",
                    avg_price,
                    executed_qty,
                    cummulative_quote_qty,
                    strategy_func.__name__,
                )
            except Exception as e:
                print(f"An error occurred while closing position on exit: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A simple Binance trading bot.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="macd",
        choices=["macd", "random"],
        help="The trading strategy to use. Default is macd.",
    )
    parser.add_argument(
        "--trade-amount",
        type=float,
        default=15.0,
        help="The amount in USDT to use for each trade. Default is 15 USDT. "
        "Note: Binance has minimum order sizes.",
    )
    parser.add_argument(
        "--max-capital",
        type=float,
        default=1000.0,
        help="The maximum available capital in USDT to use for positions. Default is 1000 USDT.",
    )
    args = parser.parse_args()

    if args.trade_amount < 11:
        # Binance minimum order size is often around 10 USDT. Setting a bit higher.
        print(
            "Warning: trade-amount is very low. Binance might reject the order. Recommended: > 11 USDT."
        )

    if args.trade_amount > args.max_capital:
        raise ValueError("trade-amount cannot be greater than max-capital.")

    strategies = {
        "macd": macd_strategy,
        "random": random_strategy,
    }
    selected_strategy = strategies[args.strategy]

    main(selected_strategy, args.trade_amount, args.max_capital)
