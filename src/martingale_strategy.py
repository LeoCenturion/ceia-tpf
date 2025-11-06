from backtest_utils import TrialStrategy, run_optimizations


class MartingaleStrategy(TrialStrategy):
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


class DelayedMartingaleStrategy(TrialStrategy):
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

class AntiMartingaleStrategy(TrialStrategy):
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

# AI implement a martingale with trend filter strategy:
# By combining the Theory of Runs with the direction of
# the trend, and the upward bias that we see in the table of
# runs, the chance of being on the correct side of the
# longest runs is increased, and the size of the price move
# in the direction of the trend may also tend to be larger.
# Long-term trends will offer the best chance of identifying
# the direction of prices. We can create rules to take
# advantage of this:
# 1.If the trend has just turned up, based on a moving
# average, enter a long position with 1 unit (the initial
# investment divided by the price).
# 2.3.4.If the trend is still up and the price closes lower,
# double the size of the long position.
# If the trend is up and the price closes higher, remove
# all positions in excess of the original 1 unit.
# If the trend turns down, exit all longs and sell short 1
# unit.
# 5.If the trend is down and prices close higher, double
# the size of the short position.
# 6.If the trend is down and prices close lower, cover all
# short positions in excess of the original 1 unit.
# Because it is possible for prices to go the wrong way
# longer than you have money, we need to cap the number
# of times we can double down. AI!
def main():
    """Main function to run the optimization with default parameters."""
    strategies = {
        # "MartingaleStrategy": MartingaleStrategy,
        # "DelayedMartingaleStrategy": DelayedMartingaleStrategy,
        "AntiMartingaleStrategy": AntiMartingaleStrategy,
    }
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2018-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Martingale Strategies",
        n_trials_per_strategy=12,
        n_jobs=12,
    )


if __name__ == "__main__":
    main()
