from src.backtesting.backtesting import TrialStrategy, run_optimizations
from src.data_analysis.data_analysis import sma

# AI refactor all init methods to be __init__, respect the parent constructor AI!

class MartingaleStrategy(TrialStrategy):  # pylint: disable=attribute-defined-outside-init
    """
    A Martingale betting strategy.

    Always bets long. If a trade is a loss, the next trade size is multiplied by a factor.
    If a trade is a win, the trade size resets to the initial amount.
    Each trade is held for one bar.
    """

    initial_trade_size = 0.01  # Percentage of equity
    multiplier = 2.0  # Multiplier for the next trade on loss

    def init(self):
        self.trade_size = self.initial_trade_size

    def next(self):
        # Close any open position from the previous bar.
        # The profit/loss of this trade determines the size of the next one.
        if self.position:
            # We assume we are always long.
            # A "win" is if the current close is higher than the entry price.
            if self.data.Close[-1] > self.trades[-1].entry_price:
                # Last trade was a win, reset trade size
                self.trade_size = self.initial_trade_size
            else:
                # Last trade was a loss, increase trade size
                self.trade_size *= self.multiplier

            self.position.close()

        # Place a new bet (buy order) for the next period, capped at 99% of equity.
        trade_size = min(self.trade_size, 0.99)
        if self.equity > 0 and trade_size > 0:
            self.buy(size=trade_size)

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            "initial_trade_size": trial.suggest_float(
                "initial_trade_size", 0.005, 0.05
            ),
            "multiplier": trial.suggest_float("multiplier", 1.5, 3.0),
        }


class DelayedMartingaleStrategy(
    TrialStrategy
):  # pylint: disable=attribute-defined-outside-init
    """
    A delayed Martingale strategy.

    Waits for a specified number of consecutive down periods before starting to bet.
    Always bets long. If a trade is a loss, the next trade size is multiplied by a factor.
    If a trade is a win, the trade size resets to the initial amount.
    Each trade is held for one bar.
    """

    initial_trade_size = 0.01  # Percentage of equity
    multiplier = 2.0  # Multiplier for the next trade on loss
    wait_periods = 3  # Number of consecutive down periods to wait for

    def init(self):
        self.trade_size = self.initial_trade_size
        self.down_periods_counter = 0

    def next(self):
        # Update consecutive down periods counter
        if len(self.data.Close) > 1 and self.data.Close[-1] < self.data.Close[-2]:
            self.down_periods_counter += 1
        else:
            self.down_periods_counter = 0

        # Close any open position from the previous bar.
        # The profit/loss of this trade determines the size of the next one.
        if self.position:
            # We assume we are always long.
            # A "win" is if the current close is higher than the entry price.
            if self.data.Close[-1] > self.trades[-1].entry_price:
                # Last trade was a win, reset trade size
                self.trade_size = self.initial_trade_size
            else:
                # Last trade was a loss, increase trade size
                self.trade_size *= self.multiplier

            self.position.close()

        # Entry logic: only bet if we have waited enough down periods
        if not self.position and self.down_periods_counter >= self.wait_periods:
            trade_size = min(self.trade_size, 0.99)
            if self.equity > 0 and trade_size > 0:
                self.buy(size=trade_size)

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            "initial_trade_size": trial.suggest_float(
                "initial_trade_size", 0.005, 0.05
            ),
            "multiplier": trial.suggest_float("multiplier", 1.5, 3.0),
            "wait_periods": trial.suggest_int("wait_periods", 2, 5),
        }


class AntiMartingaleStrategy(
    TrialStrategy
):  # pylint: disable=attribute-defined-outside-init
    """
    An anti-Martingale (reverse Martingale) betting strategy.

    Always bets long. If a trade is a win, the next trade size is multiplied by a factor.
    This continues until a loss occurs or a maximum number of consecutive wins is reached.
    If a trade is a loss, the trade size resets to the initial amount.
    Each trade is held for one bar.
    """

    initial_trade_size = 0.01  # Percentage of equity
    multiplier = 2.0  # Multiplier for the next trade on win
    max_win_streak = 5  # Maximum number of consecutive wins before resetting

    def init(self):
        self.trade_size = self.initial_trade_size
        self.win_streak = 0

    def next(self):
        # Close any open position from the previous bar.
        # The profit/loss of this trade determines the size of the next one.
        if self.position:
            # We assume we are always long.
            # A "win" is if the current close is higher than the entry price.
            if self.data.Close[-1] > self.trades[-1].entry_price:
                # Last trade was a win, increase trade size and streak
                self.win_streak += 1
                if self.win_streak >= self.max_win_streak:
                    # Max streak reached, reset
                    self.trade_size = self.initial_trade_size
                    self.win_streak = 0
                else:
                    # Continue streak, increase size
                    self.trade_size *= self.multiplier
            else:
                # Last trade was a loss, reset trade size and streak
                self.trade_size = self.initial_trade_size
                self.win_streak = 0

            self.position.close()

        # Place a new bet (buy order) for the next period, capped at 99% of equity.
        trade_size = min(self.trade_size, 0.99)
        if self.equity > 0 and trade_size > 0:
            self.buy(size=trade_size)

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            "initial_trade_size": trial.suggest_float(
                "initial_trade_size", 0.005, 0.05
            ),
            "multiplier": trial.suggest_float("multiplier", 1.5, 3.0),
            "max_win_streak": trial.suggest_int("max_win_streak", 3, 10),
        }


class MartingaleWithTrendFilter(
    TrialStrategy
):  # pylint: disable=attribute-defined-outside-init
    """
    A Martingale strategy that uses a moving average as a trend filter.
    It doubles down on positions when the price moves against the trend,
    and reduces the position to the initial size when the price moves with the trend.
    """

    ma_window = 50
    initial_trade_size = 0.01  # Percentage of equity
    max_doubles = 3

    def init(self):
        self.ma = self.I(sma, self.data.Close, self.ma_window)
        self.trend = 0  # 1 for up, -1 for down
        self.doubles_count = 0
        self.base_trade_size = 0
        self.is_sizing_trade = False

    def next(self):
        # Capture base trade size on the bar after the initial trade
        if self.is_sizing_trade and self.position:
            self.base_trade_size = abs(self.position.size)
            self.is_sizing_trade = False

        if len(self.data.Close) < self.ma_window:
            return

        new_trend = 1 if self.data.Close[-1] > self.ma[-1] else -1

        if new_trend != self.trend:
            if self.position:
                self.position.close()

            if new_trend == 1:
                self.buy(size=self.initial_trade_size)
            else:  # new_trend == -1
                self.sell(size=self.initial_trade_size)

            self.doubles_count = 0
            self.trend = new_trend
            self.is_sizing_trade = True
        else:
            if not self.position or self.is_sizing_trade:
                return

            if self.trend == 1:  # Uptrend continuation
                if self.data.Close[-1] < self.data.Close[-2]:  # Price closes lower
                    if self.doubles_count < self.max_doubles:
                        self.buy(size=self.position.size)
                        self.doubles_count += 1
                elif self.data.Close[-1] > self.data.Close[-2]:  # Price closes higher
                    if self.position.size > self.base_trade_size > 0:
                        size_to_close = self.position.size - self.base_trade_size
                        self.sell(size=size_to_close)
                        self.doubles_count = 0

            elif self.trend == -1:  # Downtrend continuation
                if self.data.Close[-1] > self.data.Close[-2]:  # Price closes higher
                    if self.doubles_count < self.max_doubles:
                        self.sell(size=abs(self.position.size))
                        self.doubles_count += 1
                elif self.data.Close[-1] < self.data.Close[-2]:  # Price closes lower
                    if abs(self.position.size) > self.base_trade_size > 0:
                        size_to_close = abs(self.position.size) - self.base_trade_size
                        self.buy(size=size_to_close)
                        self.doubles_count = 0

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            "ma_window": trial.suggest_int("ma_window", 20, 200),
            "initial_trade_size": trial.suggest_float("initial_trade_size", 0.01, 0.1),
            "max_doubles": trial.suggest_int("max_doubles", 1, 5),
        }


def main():
    """Main function to run the optimization with default parameters."""
    strategies = {
        # "MartingaleStrategy": MartingaleStrategy,
        # "DelayedMartingaleStrategy": DelayedMartingaleStrategy,
        # "AntiMartingaleStrategy": AntiMartingaleStrategy,
        "MartingaleWithTrendFilter": MartingaleWithTrendFilter,
    }
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2018-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Martingale Strategies",
        n_trials_per_strategy=50,
        n_jobs=12,
    )


if __name__ == "__main__":
    main()
