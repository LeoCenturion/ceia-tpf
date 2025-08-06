import ccxt
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from prophet import Prophet


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
        df.rename(
            columns={"date": "timestamp", "Volume BTC": "volume"}, inplace=True
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
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
    return df


# Indicator functions to be used with `self.I`
def pct_change(series):
    return pd.Series(series).pct_change()


def sma(series, n):
    return pd.Series(series).rolling(n).mean()


def ewm(series, span):
    return pd.Series(series).ewm(span=span, adjust=False).mean()


def rsi_indicator(series, n=14):
    delta = pd.Series(series).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = sma(gain, n)
    avg_loss = sma(loss, n)

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def momentum_indicator(series, window=10):
    return pd.Series(series).diff(window)


class MaCrossover(Strategy):
    short_window = 50
    long_window = 200
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        price_change = self.I(pct_change, self.data.Close)
        self.ma_short = self.I(sma, price_change, self.short_window)
        self.ma_long = self.I(sma, price_change, self.long_window)

    def next(self):
        price = self.data.Close[-1]
        if crossover(self.ma_short, self.ma_long):
            self.buy(
                sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit)
            )
        elif crossover(self.ma_long, self.ma_short):
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )


class RsiStrategy(Strategy):
    rsi_window = 14
    overbought = 70
    oversold = 30
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.rsi = self.I(rsi_indicator, self.data.Close, self.rsi_window)

    def next(self):
        price = self.data.Close[-1]
        if self.rsi > self.overbought and not self.position.is_short:
            self.position.close()
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )
        elif self.rsi < self.oversold and not self.position.is_long:
            self.position.close()
            self.buy(
                sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit)
            )


class EwmaCrossover(Strategy):
    short_span = 50
    long_span = 200
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        price_change = self.I(pct_change, self.data.Close)
        self.ewma_short = self.I(ewm, price_change, self.short_span)
        self.ewma_long = self.I(ewm, price_change, self.long_span)

    def next(self):
        price = self.data.Close[-1]
        if crossover(self.ewma_short, self.ewma_long):
            self.buy(
                sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit)
            )
        elif crossover(self.ewma_long, self.ewma_short):
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )


class MomentumStrategy(Strategy):
    window = 10
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.momentum = self.I(momentum_indicator, self.data.Close, self.window)

    def next(self):
        price = self.data.Close[-1]
        if self.momentum > 0 and not self.position.is_long:
            self.position.close()
            self.buy(
                sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit)
            )
        elif self.momentum < 0 and not self.position.is_short:
            self.position.close()
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )


class ProphetStrategy(Strategy):
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        # This is a simplistic implementation and has a strong
        # lookahead bias as it fits the Prophet model on the entire dataset.
        df_prophet = self.data.df.reset_index().rename(
            columns={"timestamp": "ds", "Close": "y"}
        )
        df_prophet["y"] = df_prophet["y"].pct_change()
        df_prophet_train = df_prophet.dropna()

        model = Prophet()
        model.fit(df_prophet_train)

        future = df_prophet[["ds"]]
        forecast = model.predict(future)
        forecast.set_index("ds", inplace=True)

        data_with_forecast = self.data.df.join(forecast[["trend"]])
        data_with_forecast["price_change"] = data_with_forecast["Close"].pct_change()

        signals = pd.Series(index=data_with_forecast.index, dtype=float)
        signals[:] = 0
        signals.loc[
            data_with_forecast["price_change"] > data_with_forecast["trend"]
        ] = 1
        signals.loc[
            data_with_forecast["price_change"] < data_with_forecast["trend"]
        ] = -1
        self.signals = signals.fillna(0).values

    def next(self):
        price = self.data.Close[-1]
        current_index = len(self.data) - 1
        if current_index >= len(self.signals):
            return

        signal = self.signals[current_index]
        if signal == 1 and not self.position.is_long:
            self.buy(
                sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit)
            )
        elif signal == -1 and not self.position.is_short:
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )


def main():
    # Fetch historical data
    historical_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2018-01-01",
        end_date="2024-01-01",
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
    )

    strategies = {
        "MA Crossover": MaCrossover,
        "RSI": RsiStrategy,
        "EWMA Crossover": EwmaCrossover,
        "Momentum": MomentumStrategy,
        "Prophet": ProphetStrategy,
    }

    for name, strat_class in strategies.items():
        bt = Backtest(
            historical_data,
            strat_class,
            cash=10000,
            commission=0.0005,
            margin=1 / 5.0,
        )
        stats = bt.run()
        print(f"------ {name} ------")
        print(stats)
        # For a single plot, uncomment the following line and run one strategy
        # bt.plot()


if __name__ == "__main__":
    main()

# Requirements:
# pip install backtesting ccxt pandas prophet
