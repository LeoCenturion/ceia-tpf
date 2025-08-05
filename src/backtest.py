import os
import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
from prophet import Prophet
from typing import List, Dict, Callable


class TradingBacktester:
    def __init__(
        self,
        initial_balance: float = 10000,
        commission_rate: float = 0.0005,  # 0.05% Binance futures fee
        leverage: float = 1,
    ):
        """
        Initialize the backtester with trading parameters

        :param initial_balance: Starting capital
        :param commission_rate: Trading commission rate
        :param leverage: Trading leverage
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.commission_rate = commission_rate
        self.leverage = leverage

        # Store backtest results
        self.trades = []
        self.equity_curve = []

    def fetch_historical_data(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        start_date: str = None,
        end_date: str = None,
        data_path: str = None,
    ) -> pd.DataFrame:
        """
        Fetch historical price data for backtesting

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

            return df

        exchange = ccxt.binance(
            {"enableRateLimit": True, "options": {"defaultType": "future"}}
        )
        start_timestamp = None
        # Convert timestamps if dates provided
        if start_date:
            start_timestamp = exchange.parse8601(start_date)
        else:
            start_timestamp = None
        end_timestamp = None
        if end_date:
            end_timestamp = exchange.parse8601(end_date)
        else:
            end_timestamp = None

        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe,
            since=start_timestamp,
            limit=end_timestamp,  # Adjust as needed
        )

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df

    def moving_average_crossover_strategy(
        self, data: pd.DataFrame, short_window: int = 50, long_window: int = 200
    ) -> pd.Series:
        """
        Moving Average Crossover strategy

        :param data: Historical price data
        :param short_window: Short-term moving average window
        :param long_window: Long-term moving average window
        :return: Trading signals
        """
        signals = pd.Series(index=data.index, dtype=float)
        signals[:] = 0

        price_change = data["close"].pct_change()

        data["MA_short"] = price_change.rolling(window=short_window).mean()
        data["MA_long"] = price_change.rolling(window=long_window).mean()

        # Generate signals
        signals.loc[data["MA_short"] > data["MA_long"]] = 1  # Long
        signals.loc[data["MA_short"] < data["MA_long"]] = -1  # Short

        return signals

    def rsi_strategy(
        self,
        data: pd.DataFrame,
        rsi_window: int = 14,
        overbought: float = 70,
        oversold: float = 30,
    ) -> pd.Series:
        """
        RSI (Relative Strength Index) trading strategy

        :param data: Historical price data
        :param rsi_window: RSI calculation window
        :param overbought: Overbought threshold
        :param oversold: Oversold threshold
        :return: Trading signals
        """
        signals = pd.Series(index=data.index, dtype=float)
        signals[:] = 0

        # Calculate RSI
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signals.loc[rsi > overbought] = -1  # Sell signal
        signals.loc[rsi < oversold] = 1  # Buy signal

        return signals

    def ewma_crossover_strategy(
        self, data: pd.DataFrame, short_span: int = 50, long_span: int = 200
    ) -> pd.Series:
        """
        Exponential Moving Average (EWMA) Crossover strategy

        :param data: Historical price data
        :param short_span: Short-term EWMA span
        :param long_span: Long-term EWMA span
        :return: Trading signals
        """
        signals = pd.Series(index=data.index, dtype=float)
        signals[:] = 0

        price_change = data["close"].pct_change()

        data["EWMA_short"] = price_change.ewm(span=short_span, adjust=False).mean()
        data["EWMA_long"] = price_change.ewm(span=long_span, adjust=False).mean()

        # Generate signals
        signals.loc[data["EWMA_short"] > data["EWMA_long"]] = 1  # Long
        signals.loc[data["EWMA_short"] < data["EWMA_long"]] = -1  # Short

        return signals

    def momentum_strategy(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """
        Simple momentum strategy.

        :param data: Historical price data
        :param window: Momentum calculation window
        :return: Trading signals
        """
        signals = pd.Series(index=data.index, dtype=float)
        signals[:] = 0

        momentum = data["close"].diff(window)

        # Generate signals
        signals.loc[momentum > 0] = 1  # Buy signal
        signals.loc[momentum < 0] = -1  # Sell signal

        return signals

    def prophet_strategy(self, data: pd.DataFrame) -> pd.Series:
        """
        Prophet-based trading strategy.

        .. warning::
            This is a simplistic implementation and has a strong
            lookahead bias as it fits the Prophet model on the entire dataset.
            A proper implementation would use a rolling forecast to avoid
            using future data. This is for demonstration purposes only.

        :param data: Historical price data.
        :return: Trading signals.
        """
        signals = pd.Series(index=data.index, dtype=float)
        signals[:] = 0

        # Prepare data for Prophet using price changes
        prophet_df = data.reset_index()[["timestamp", "close"]].rename(
            columns={"timestamp": "ds", "close": "y"}
        )
        prophet_df["y"] = prophet_df["y"].pct_change()
        prophet_df_train = prophet_df.dropna()

        model = Prophet()
        model.fit(prophet_df_train)

        # Predict on all dates
        future = prophet_df[["ds"]]
        forecast = model.predict(future)

        # Merge forecast back to original dataframe
        forecast.set_index("ds", inplace=True)
        data_with_forecast = data.join(forecast[["trend"]])
        data_with_forecast["price_change"] = data["close"].pct_change()

        # Generate signals based on whether the price change is above or below the trend
        signals.loc[
            data_with_forecast["price_change"] > data_with_forecast["trend"]
        ] = 1
        signals.loc[
            data_with_forecast["price_change"] < data_with_forecast["trend"]
        ] = -1

        return signals

    def backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
    ) -> Dict:
        """
        Backtest trading strategy

        :param data: Historical price data
        :param signals: Trading signals
        :param stop_loss: Stop loss percentage
        :param take_profit: Take profit percentage
        :return: Backtest results
        """
        position = 0
        entry_price = 0
        trades = []
        balance_history = [self.initial_balance]

        for i in range(len(data)):
            current_price = data["close"].iloc[i]

            # Close position based on stop loss or take profit
            if position != 0:
                if position > 0:  # Long position
                    if (current_price / entry_price - 1) <= -stop_loss or (
                        current_price / entry_price - 1
                    ) >= take_profit:
                        profit = (current_price - entry_price) * abs(position)
                        balance_history.append(balance_history[-1] + profit)
                        trades.append(
                            {
                                "type": "long_close",
                                "price": current_price,
                                "profit": profit,
                            }
                        )
                        position = 0
                else:  # Short position
                    if (entry_price / current_price - 1) <= -stop_loss or (
                        entry_price / current_price - 1
                    ) >= take_profit:
                        profit = (entry_price - current_price) * abs(position)
                        balance_history.append(balance_history[-1] + profit)
                        trades.append(
                            {
                                "type": "short_close",
                                "price": current_price,
                                "profit": profit,
                            }
                        )
                        position = 0

            # Enter new position
            if signals.iloc[i] == 1 and position <= 0:  # Go long
                if position < 0:  # Close short first
                    profit = (entry_price - current_price) * abs(position)
                    balance_history.append(balance_history[-1] + profit)
                    trades.append(
                        {
                            "type": "short_close",
                            "price": current_price,
                            "profit": profit,
                        }
                    )

                position_size = balance_history[-1] * self.leverage / current_price
                entry_price = current_price
                position = position_size
                trades.append({"type": "long_open", "price": current_price})

            elif signals.iloc[i] == -1 and position >= 0:  # Go short
                if position > 0:  # Close long first
                    profit = (current_price - entry_price) * abs(position)
                    balance_history.append(balance_history[-1] + profit)
                    trades.append(
                        {"type": "long_close", "price": current_price, "profit": profit}
                    )

                position_size = balance_history[-1] * self.leverage / current_price
                entry_price = current_price
                position = -position_size
                trades.append({"type": "short_open", "price": current_price})

        # Final position closing
        if position > 0:
            profit = (data["close"].iloc[-1] - entry_price) * abs(position)
            balance_history.append(balance_history[-1] + profit)
            trades.append(
                {
                    "type": "long_close",
                    "price": data["close"].iloc[-1],
                    "profit": profit,
                }
            )
        elif position < 0:
            profit = (entry_price - data["close"].iloc[-1]) * abs(position)
            balance_history.append(balance_history[-1] + profit)
            trades.append(
                {
                    "type": "short_close",
                    "price": data["close"].iloc[-1],
                    "profit": profit,
                }
            )

        # Calculate performance metrics
        total_return = (
            (balance_history[-1] - self.initial_balance) / self.initial_balance * 100
        )

        return {
            "final_balance": balance_history[-1],
            "total_return_percent": total_return,
            "trades": trades,
            "balance_history": balance_history,
        }

    def plot_performance(self, balance_history: List[float]):
        """
        Plot equity curve

        :param balance_history: List of balance values over time
        """
        plt.figure(figsize=(12, 6))
        plt.plot(balance_history)
        plt.title("Equity Curve")
        plt.xlabel("Trades")
        plt.ylabel("Balance")
        plt.tight_layout()
        plt.show()


def main():
    # Initialize backtester
    backtester = TradingBacktester(
        initial_balance=10000, commission_rate=0.0005, leverage=5
    )

    # Fetch historical data
    historical_data = backtester.fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2018-01-01",
        end_date="2024-01-01",
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
    )

    # Generate signals using different strategies
    ma_signals = backtester.moving_average_crossover_strategy(
        historical_data, short_window=50, long_window=200
    )

    rsi_signals = backtester.rsi_strategy(
        historical_data, rsi_window=14, overbought=70, oversold=30
    )

    ewma_signals = backtester.ewma_crossover_strategy(
        historical_data, short_span=50, long_span=200
    )

    momentum_signals = backtester.momentum_strategy(historical_data, window=10)

    prophet_signals = backtester.prophet_strategy(historical_data)

    # Backtest strategies
    ma_results = backtester.backtest(
        historical_data, ma_signals, stop_loss=0.05, take_profit=0.10
    )

    rsi_results = backtester.backtest(
        historical_data, rsi_signals, stop_loss=0.05, take_profit=0.10
    )

    ewma_results = backtester.backtest(
        historical_data, ewma_signals, stop_loss=0.05, take_profit=0.10
    )

    momentum_results = backtester.backtest(
        historical_data, momentum_signals, stop_loss=0.05, take_profit=0.10
    )

    prophet_results = backtester.backtest(
        historical_data, prophet_signals, stop_loss=0.05, take_profit=0.10
    )

    # Print results
    print("Moving Average Crossover Strategy:")
    print(f"Final Balance: ${ma_results['final_balance']:.2f}")
    print(f"Total Return: {ma_results['total_return_percent']:.2f}%")

    print("\nRSI Strategy:")
    print(f"Final Balance: ${rsi_results['final_balance']:.2f}")
    print(f"Total Return: {rsi_results['total_return_percent']:.2f}%")

    print("\nEWMA Crossover Strategy:")
    print(f"Final Balance: ${ewma_results['final_balance']:.2f}")
    print(f"Total Return: {ewma_results['total_return_percent']:.2f}%")

    print("\nMomentum Strategy:")
    print(f"Final Balance: ${momentum_results['final_balance']:.2f}")
    print(f"Total Return: {momentum_results['total_return_percent']:.2f}%")

    print("\nProphet Strategy:")
    print(f"Final Balance: ${prophet_results['final_balance']:.2f}")
    print(f"Total Return: {prophet_results['total_return_percent']:.2f}%")

    # Plot performance
    # backtester.plot_performance(ma_results['balance_history'])
    # backtester.plot_performance(rsi_results['balance_history'])


if __name__ == "__main__":
    main()

# Requirements:
# pip install ccxt pandas numpy matplotlib
