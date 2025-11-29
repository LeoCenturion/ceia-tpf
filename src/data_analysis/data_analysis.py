"""Data fetching and processing utilities."""
import ccxt
import pandas as pd


def fetch_historical_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = None,
    end_date: str = None,
    data_path: str = None,
) -> pd.DataFrame:
    """
    Fetch historical price data for backtesting.

    The `backtesting` library requires column names: 'Open', 'High', 'Low', 'Close'.

    :param symbol: Trading pair
    :param timeframe: Candle timeframe
    :param start_date: Start date for data fetch
    :param end_date: End date for data fetch
    :param data_path: Path to local CSV file. If provided, data is loaded from here.
    :return: DataFrame with OHLCV data
    """
    if data_path:
        df = pd.read_csv(data_path)
        df.rename(columns={"date": "timestamp", "Volume BTC": "volume"}, inplace=True)
        # s_timestamp = str(int(df['unix'].iloc[0]))
        # unit = 'us' if len(s_timestamp) > 13 else 'ms'
        # df["timestamp"] = pd.to_datetime(df["unix"], unit=unit)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        if start_date:
            start_date = pd.to_datetime(start_date)
            if start_date.tzinfo is not None:
                start_date = start_date.tz_convert(None)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            if end_date.tzinfo is not None:
                end_date = end_date.tz_convert(None)
            df = df[df.index <= end_date]

    else:
        exchange = ccxt.binance(
            {"enableRateLimit": True, "options": {"defaultType": "future"}}
        )
        start_timestamp = None
        if start_date:
            start_timestamp = exchange.parse8601(start_date)
        else:
            start_timestamp = None
        end_timestamp = None
        if end_date:
            end_timestamp = exchange.parse8601(end_date)
        else:
            end_timestamp = None

        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe,
            since=start_timestamp,
            limit=end_timestamp,
        )

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

    # The backtesting library requires uppercase column names
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    df.sort_index(inplace=True)
    return df


# Indicator functions to be used with `self.I`
def pct_change(series):
    """Calculate the percentage change of a series."""
    return pd.Series(series).pct_change()


def sma(series, n):
    """Calculate the simple moving average of a series."""
    return pd.Series(series).rolling(n).mean()


def ewm(series, span):
    """Calculate the exponential moving average of a series."""
    return pd.Series(series).ewm(span=span, adjust=False).mean()


def std(series, n):
    """Calculate the rolling standard deviation of a series."""
    return pd.Series(series).rolling(n).std()


def adjust_data_to_ubtc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the price data from BTC to micro-BTC (uBTC).
    1 BTC = 1,000,000 uBTC.
    This function divides the OHLC prices by 1,000,000.
    """
    df_copy = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col] / 1_000_000
    return df_copy
