import mlflow
from backtesting import Strategy
from backtesting.lib import crossover

from backtest_utils import (
    pct_change,
    sma,
    ewm,
    std,
    fetch_historical_data,
    adjust_data_to_ubtc,
    optimize_strategy,
)


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
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif crossover(self.ma_long, self.ma_short):
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            'short_window': trial.suggest_int('short_window', 10, 100),
            'long_window': trial.suggest_int('long_window', 50, 250),
        }


class BollingerBands(Strategy):
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
            'bb_window': trial.suggest_int('bb_window', 10, 100),
            'bb_std': trial.suggest_float('bb_std', 1.5, 3.5),
        }


class MACD(Strategy):
    fast_span = 12
    slow_span = 26
    signal_span = 9
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.macd = self.I(ewm, self.data.Close, self.fast_span) - self.I(ewm, self.data.Close, self.slow_span)
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
            'fast_span': trial.suggest_int('fast_span', 5, 50),
            'slow_span': trial.suggest_int('slow_span', 20, 100),
            'signal_span': trial.suggest_int('signal_span', 5, 50),
        }

#AI move this to backtest_utils.py as a utility AI!
def run_optimizations(strategies, data_path, start_date, tracking_uri, experiment_name, n_trials_per_strategy=10):
    """
    Run optimization for a set of strategies.

    :param strategies: Dictionary of strategy names to strategy classes.
    :param data_path: Path to the historical data CSV file.
    :param start_date: Start date for the data.
    :param tracking_uri: MLflow tracking URI.
    :param experiment_name: MLflow experiment name.
    :param n_trials_per_strategy: Number of Optuna trials for each strategy.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    data = fetch_historical_data(
        data_path=data_path,
        start_date=start_date
    )
    data = adjust_data_to_ubtc(data)

    for name, strategy in strategies.items():
        # This outer run is for grouping the optimization trials
        with mlflow.start_run(run_name=f"Optimize_{name}"):
            print(f"Optimizing {name}...")
            optimize_strategy(data, strategy, n_trials=n_trials_per_strategy, study_name=name)
            print(f"Optimization for {name} complete.")

def main():
    """Main function to run the optimization with default parameters."""
    strategies = {
        "MaCrossover": MaCrossover,
        "BollingerBands": BollingerBands,
        "MACD": MACD
    }
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Trading Strategies",
        n_trials_per_strategy=10
    )


if __name__ == "__main__":
    main()
