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


def main():
    """Main function to run the optimization with default parameters."""
    strategies = {"MartingaleStrategy": MartingaleStrategy}
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2018-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Martingale Strategies",
        n_trials_per_strategy=20,
        n_jobs=12,
    )


if __name__ == "__main__":
    main()
