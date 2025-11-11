from functools import partial

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import trange

from backtest_utils import fetch_historical_data
from chronos import Chronos2Pipeline

# Initialize Chronos2 Pipeline globally to avoid reloading the model in each trial
# Using device_map="auto" will automatically use a GPU if available.
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="auto")


def objective(trial: optuna.Trial, data: pd.DataFrame, test_size: float, step_size: int, prediction_length: int) -> float:
    """
    Optuna objective function for Chronos2 model.
    Performs a walk-forward validation and returns the Root Mean Squared Error.
    """
    # === 1. Define Hyperparameter Search Space ===
    context_length = trial.suggest_int('context_length', 64, 512, step=32)
    num_samples = trial.suggest_int('num_samples', 10, 40)
    temperature = trial.suggest_float('temperature', 0.5, 1.0)
    top_k = trial.suggest_int('top_k', 20, 50)
    top_p = trial.suggest_float('top_p', 0.8, 1.0)

    # === 2. Prepare data for Chronos2 format ===
    # Chronos2 requires columns: id, timestamp, target
    data_chronos = data.copy()
    data_chronos = data_chronos.reset_index()  # move timestamp from index to column
    data_chronos.rename(columns={'timestamp': 'timestamp', 'close': 'target'}, inplace=True)
    data_chronos['id'] = 'BTC/USDT'

    # === 3. Walk-Forward Validation ===
    split_index = int(len(data_chronos) * (1 - test_size))
    predictions = []
    actuals = []

    print(f"Starting walk-forward validation for Trial {trial.number}...")
    for i in trange(split_index, len(data_chronos) - prediction_length, step_size, desc=f"Trial {trial.number}"):
        # Define context (training) and forecast (testing) windows
        context_end = i
        context_start = max(0, context_end - context_length)
        forecast_end = i + prediction_length

        context_df = data_chronos.iloc[context_start:context_end]

        if len(context_df) < 2:  # Need at least 2 data points for prediction
            continue

        # Generate predictions
        try:
            pred_df = pipeline.predict_df(
                context_df,
                prediction_length=prediction_length,
                num_samples=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                quantile_levels=[0.5],  # We only need the median for point forecast
                id_column="id",
                timestamp_column="timestamp",
                target_column="target",
            )
        except Exception as e:
            print(f"Prediction failed at step {i} with error: {e}. Skipping step.")
            continue

        forecast_actual_df = data_chronos.iloc[i:forecast_end]
        merged_df = pd.merge(forecast_actual_df[['timestamp', 'target']], pred_df[['timestamp', '0.5']], on='timestamp', how='inner')

        if not merged_df.empty:
            predictions.extend(merged_df['0.5'].values)
            actuals.extend(merged_df['target'].values)

    if not actuals:
        print("No predictions were made. Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    # === 4. Calculate Metric ===
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)

    print(f"Trial {trial.number} finished. RMSE: {rmse:.4f}")

    return rmse


def run_study(data, study_name_in_db, n_trials):
    """
    Runs an Optuna study for the Chronos2 model and logs results to MLflow.
    """
    # --- MLflow setup ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(study_name_in_db)

    # --- Walk-forward validation parameters ---
    TEST_SIZE = 0.3  # 30% of data for testing
    STEP_SIZE = 24  # Refit and predict every 24 hours
    PREDICTION_LENGTH = 24  # Forecast 24 hours into the future

    # --- Optuna Study ---
    db_file_name = "optuna-study"
    storage_name = f"sqlite:///{db_file_name}.db"

    print(f"Starting Optuna study: '{study_name_in_db}'. Storage: {storage_name}")

    objective_with_data = partial(objective, data=data, test_size=TEST_SIZE, step_size=STEP_SIZE, prediction_length=PREDICTION_LENGTH)

    study = optuna.create_study(
        direction='minimize',  # We want to minimize RMSE
        study_name=study_name_in_db,
        storage=storage_name,
        load_if_exists=True
    )

    def mlflow_callback(study, trial):
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_params(trial.params)
            mlflow.log_metric("rmse", trial.value)

    study.optimize(objective_with_data, n_trials=n_trials, callbacks=[mlflow_callback])

    # --- Log Best Results ---
    print("\n--- Optuna Study Best Results ---")
    try:
        best_trial = study.best_trial
        print(f"Best trial value (RMSE): {best_trial.value}")
        print("Best parameters found:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        # Log best trial to a parent MLflow run
        with mlflow.start_run(run_name="best_trial_summary"):
            mlflow.log_params(best_trial.params)
            mlflow.log_metric("best_rmse", best_trial.value)

    except ValueError:
        print("No successful trials were completed.")


def main():
    """
    Main function to load data and run the Optuna study.
    """
    N_TRIALS = 50
    STUDY_NAME = 'chronos2_btc_price_prediction'

    print("Loading historical data...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z"
    )

    run_study(data, STUDY_NAME, N_TRIALS)


if __name__ == "__main__":
    main()
