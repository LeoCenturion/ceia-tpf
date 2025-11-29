from backtesting.lib import crossover

from backtesting.backtesting import TrialStrategy, run_optimizations
from data_analysis.data_analysis import ewm, pct_change, sma, std
from data_analysis.indicators import rsi_indicator


class MaCrossover(TrialStrategy):  # pylint: disable=attribute-defined-outside-init
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
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif crossover(self.ma_long, self.ma_short):
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            "short_window": trial.suggest_int("short_window", 10, 100),
            "long_window": trial.suggest_int("long_window", 50, 250),
        }


class BollingerBands(TrialStrategy):  # pylint: disable=attribute-defined-outside-init
    bb_window = 20
    bb_std = 2
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.ma = self.I(sma, self.data.Close, self.bb_window)
        self.std = self.I(std, self.data.Close, self.bb_window)
        self.upper_band = self.ma + self.bb_std * self.std
        self.lower_band = self.ma - self.bb_std * self.std

    def next(self):
        price = self.data.Close[-1]
        if price < self.lower_band:
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif price > self.upper_band:
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            "bb_window": trial.suggest_int("bb_window", 10, 200, step=10),
            "bb_std": trial.suggest_float("bb_std", 0.1, 3.5, step=0.1),
        }


class MACD(TrialStrategy):  # pylint: disable=attribute-defined-outside-init
    fast_span = 12
    slow_span = 26
    signal_span = 9
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.macd = self.I(ewm, self.data.Close, self.fast_span) - self.I(
            ewm, self.data.Close, self.slow_span
        )
        self.signal = self.I(ewm, self.macd, self.signal_span)

    def next(self):
        price = self.data.Close[-1]
        if crossover(self.macd, self.signal):
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif crossover(self.signal, self.macd):
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            "fast_span": trial.suggest_int("fast_span", 5, 50),
            "slow_span": trial.suggest_int("slow_span", 20, 200, step=5),
            "signal_span": trial.suggest_int("signal_span", 5, 50),
        }


class RSIDivergence(TrialStrategy):  # pylint: disable=attribute-defined-outside-init
    rsi_window = 14
    divergence_period = 30
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.rsi = self.I(rsi_indicator, self.data.Close, self.rsi_window)

    def next(self):
        if len(self.data.Close) < self.divergence_period + 1:
            return

        price = self.data.Close[-1]

        # Bullish divergence: price makes a lower low, RSI makes a higher low
        price_low_lookback = self.data.Low[-self.divergence_period : -1]
        prev_price_low_idx_in_slice = price_low_lookback.argmin()
        prev_low_idx = -(self.divergence_period - prev_price_low_idx_in_slice)

        # Bearish divergence: price makes a higher high, RSI makes a lower high
        price_high_lookback = self.data.High[-self.divergence_period : -1]
        prev_price_high_idx_in_slice = price_high_lookback.argmax()
        prev_high_idx = -(self.divergence_period - prev_price_high_idx_in_slice)

        if (
            self.data.Low[-1] < self.data.Low[prev_low_idx]
            and self.rsi[-1] > self.rsi[prev_low_idx]
        ):
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif (
            self.data.High[-1] > self.data.High[prev_high_idx]
            and self.rsi[-1] < self.rsi[prev_high_idx]
        ):
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            "rsi_window": trial.suggest_int("rsi_window", 5, 30),
            "divergence_period": trial.suggest_int("divergence_period", 10, 60),
        }


class MultiIndicatorStrategy(
    TrialStrategy
):  # pylint: disable=attribute-defined-outside-init
    bb_window = 20
    bb_std = 2
    fast_sma_window = 50
    slow_sma_window = 100
    trailing_sl_pct = 0.05  # 5% trailing stop loss

    def init(self):
        # Bollinger Bands
        self.bb_ma = self.I(sma, self.data.Close, self.bb_window)
        self.bb_std_dev = self.I(std, self.data.Close, self.bb_window)
        self.upper_band = self.bb_ma + self.bb_std * self.bb_std_dev
        self.lower_band = self.bb_ma - self.bb_std * self.bb_std_dev

        # SMAs for trend confirmation
        self.sma_fast = self.I(sma, self.data.Close, self.fast_sma_window)
        self.sma_slow = self.I(sma, self.data.Close, self.slow_sma_window)

    def next(self):
        price = self.data.Close[-1]

        # Trailing stop-loss logic
        for trade in self.trades:
            if trade.is_long:
                trade.sl = max(trade.sl or 0, price * (1 - self.trailing_sl_pct))
            else:
                trade.sl = min(
                    trade.sl or float("inf"), price * (1 + self.trailing_sl_pct)
                )

        # Entry logic
        if not self.position:
            # Long Entry: Price above upper BB & uptrend confirmed by SMAs
            if price > self.upper_band and self.sma_fast > self.sma_slow:
                sl = price * (1 - self.trailing_sl_pct)
                self.buy(sl=sl)

            # Short Entry: Price below lower BB & downtrend confirmed by SMAs
            elif price < self.lower_band and self.sma_fast < self.sma_slow:
                sl = price * (1 + self.trailing_sl_pct)
                self.sell(sl=sl)

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            "bb_window": trial.suggest_int("bb_window", 10, 50),
            "bb_std": trial.suggest_float("bb_std", 0.1, 3.5, step=0.1),
            "fast_sma_window": trial.suggest_int("fast_sma_window", 20, 70),
            "slow_sma_window": trial.suggest_int("slow_sma_window", 80, 220),
            "trailing_sl_pct": trial.suggest_float("trailing_sl_pct", 0.01, 0.1),
        }


class SwingTrading(TrialStrategy):  # pylint: disable=attribute-defined-outside-init
    swing_filter_p = 0.025
    trade_mode = "aggressive"  # 'aggressive' or 'conservative'

    def init(self):
        self.swing_direction = 0  # 1 for up, -1 for down
        self.current_high = 0.0
        self.current_low = float("inf")
        self.swing_highs = []
        self.swing_lows = []

        # For conservative mode, to prevent re-entry in same swing
        self.entry_signal_triggered = False

    def next(self):
        if len(self.data) < 2:
            return

        # Initialize on first valid bar
        if self.swing_direction == 0:
            if self.data.High[-1] > self.data.High[-2]:
                self.swing_direction = 1
                self.swing_lows.append(self.data.Low[-2])
                self.current_high = self.data.High[-1]
            elif self.data.Low[-1] < self.data.Low[-2]:
                self.swing_direction = -1
                self.swing_highs.append(self.data.High[-2])
                self.current_low = self.data.Low[-1]
            return

        swing_filter = self.swing_filter_p * self.data.Close[-1]

        # --- DOWNSWING LOGIC ---
        if self.swing_direction == -1:
            # 2. Test if downswing continues
            if self.data.Low[-1] < self.current_low:
                self.current_low = self.data.Low[-1]
                # Conservative sell logic: check for breakdown
                if (
                    self.trade_mode == "conservative"
                    and len(self.swing_lows) >= 2
                    and not self.entry_signal_triggered
                    and self.current_low < self.swing_lows[-2]
                ):
                    self.sell()
                    self.entry_signal_triggered = True

            # 3. Test if downswing reverses to upswing
            if self.data.High[-1] - self.current_low > swing_filter:
                self.swing_direction = 1
                self.swing_lows.append(self.current_low)
                self.current_high = self.data.High[-1]
                self.entry_signal_triggered = False  # reset for new swing

                if self.position.is_short:
                    self.position.close()
                if self.trade_mode == "aggressive":
                    self.buy()

        # --- UPSWING LOGIC ---
        elif self.swing_direction == 1:
            # 4. Test if upswing continues
            if self.data.High[-1] > self.current_high:
                self.current_high = self.data.High[-1]
                # Conservative buy logic: check for breakout
                if (
                    self.trade_mode == "conservative"
                    and len(self.swing_highs) >= 2
                    and not self.entry_signal_triggered
                    and self.current_high > self.swing_highs[-2]
                ):
                    self.buy()
                    self.entry_signal_triggered = True

            # 5. Test if upswing reverses to downswing
            if self.current_high - self.data.Low[-1] > swing_filter:
                self.swing_direction = -1
                self.swing_highs.append(self.current_high)
                self.current_low = self.data.Low[-1]
                self.entry_signal_triggered = False  # reset for new swing

                if self.position.is_long:
                    self.position.close()
                if self.trade_mode == "aggressive":
                    self.sell()

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "swing_filter_p": trial.suggest_float("swing_filter_p", 0.01, 0.15),
            "trade_mode": trial.suggest_categorical(
                "trade_mode", ["aggressive", "conservative"]
            ),
        }


def main():
    """Main function to run the optimization with default parameters."""
    strategies = {
        # "MaCrossover": MaCrossover,
        # "BollingerBands": BollingerBands,
        # "MACD": MACD,
        # "RSIDivergence": RSIDivergence,
        "MultiIndicatorStrategy": MultiIndicatorStrategy,
        # "SwingTrading": SwingTrading
    }
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2020-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Trading Strategies",
        n_trials_per_strategy=1,
        n_jobs=12,
    )


if __name__ == "__main__":
    main()
