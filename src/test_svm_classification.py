from backtest_utils import run_classification_optimizations
from ml_strategies import SVCStrategy


def main():
    """Main function to run optimization for SVCStrategy."""
    strategies = {
        "SVCStrategy": SVCStrategy,
    }
    run_classification_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Test SVM Classification",
        n_trials_per_strategy=10,
        n_jobs=4
    )


if __name__ == "__main__":
    main()
