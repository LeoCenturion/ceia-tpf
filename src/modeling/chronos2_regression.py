from functools import partial

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, classification_report
from tqdm import trange

from src.data_analysis import fetch_historical_data
from chronos import Chronos2Pipeline

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize Chronos2 Pipeline globally to avoid reloading the model in each trial
# Using device_map="auto" is not supported, so we explicitly set the device.
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)


def objective(
    trial: optuna.Trial,
    data: pd.DataFrame,
    test_size: float,
    step_size: int,
    prediction_length: int,
) -> float:
    """
    Optuna objective function for Chronos2 model.
    Performs a walk-forward validation and returns the Root Mean Squared Error.
    """
    # === 1. Define Hyperparameter Search Space ===
    context_length = trial.suggest_int("context_length", 64, 512, step=32)
    num_samples = trial.suggest_int("num_samples", 10, 40)
    temperature = trial.suggest_float("temperature", 0.5, 1.0)
    top_k = trial.suggest_int("top_k", 20, 50)
    top_p = trial.suggest_float("top_p", 0.8, 1.0)

    # === 2. Prepare data for Chronos2 format ===
    # Chronos2 requires columns: id, timestamp, target
    data_chronos = data.copy()
    data_chronos = data_chronos.reset_index()  # move timestamp from index to column
    data_chronos.rename(
        columns={"timestamp": "timestamp", "Close": "target"}, inplace=True
    )
    data_chronos["id"] = "BTC/USDT"

    # === 3. Walk-Forward Validation ===
    split_index = int(len(data_chronos) * (1 - test_size))
    predictions = []
    actuals = []

    print(f"Starting walk-forward validation for Trial {trial.number}...")
    for i in trange(
        split_index,
        len(data_chronos) - prediction_length,
        step_size,
        desc=f"Trial {trial.number}",
    ):
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
        merged_df = pd.merge(
            forecast_actual_df[["timestamp", "target"]],
            pred_df[["timestamp", "0.5"]],
            on="timestamp",
            how="inner",
        )

        if not merged_df.empty:
            predictions.extend(merged_df["0.5"].values)
            actuals.extend(merged_df["target"].values)

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

    objective_with_data = partial(
        objective,
        data=data,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        prediction_length=PREDICTION_LENGTH,
    )

    study = optuna.create_study(
        direction="minimize",  # We want to minimize RMSE
        study_name=study_name_in_db,
        storage=storage_name,
        load_if_exists=True,
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


def train_and_predict_direction_split(data: pd.DataFrame, context_length: int = 512):
    """
    Trains the Chronos2 model to predict the direction of price change (up/down)
    on a 70/30 split using a rolling window forecast.
    """
    print("\n--- Predicting Price Direction on a 70/30 Split ---")

    # === 1. Prepare data ===
    data_chronos = data.copy()
    data_chronos = data_chronos.reset_index()
    data_chronos.rename(
        columns={"timestamp": "timestamp", "Close": "target"}, inplace=True
    )
    data_chronos["id"] = "BTC/USDT"

    # === 2. Split data ===
    split_index = int(len(data_chronos) * 0.7)
    context_df = data_chronos.iloc[max(0, split_index - context_length) : split_index]
    actual_df = data_chronos.iloc[split_index:]

    # === 3. Rolling Window Prediction for Direction ===
    print("Generating directional predictions using a rolling window...")
    y_true = []
    y_pred = []
    current_context_df = context_df.copy()

    for i in trange(len(actual_df), desc="Rolling Directional Prediction"):
        if len(current_context_df) < 2:
            continue

        last_known_price = current_context_df["target"].iloc[-1]

        try:
            # Predict one step ahead
            single_pred_df = pipeline.predict_df(
                current_context_df,
                prediction_length=1,
                quantile_levels=[0.5],  # Only need median for point forecast
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )

            predicted_price = single_pred_df["0.5"].iloc[0]

            # Convert price prediction to directional prediction (+1 for up, -1 for down)
            predicted_direction = 1 if predicted_price > last_known_price else -1
            y_pred.append(predicted_direction)

            # Determine actual direction
            actual_price = actual_df["target"].iloc[i]
            actual_direction = 1 if actual_price > last_known_price else -1
            y_true.append(actual_direction)

            # Update context for the next step with the true value
            next_actual_step = actual_df.iloc[[i]]
            current_context_df = pd.concat(
                [current_context_df, next_actual_step], ignore_index=True
            )

        except Exception as e:
            print(f"Prediction failed at step {i + 1} with error: {e}. Stopping.")
            break

    if not y_true:
        print("No predictions were made.")
        return

    # === 4. Evaluate and Print Classification Report ===
    print("\n--- Classification Report ---")
    target_names = ["Down (-1)", "Up (+1)"]
    labels = [-1, 1]
    print(
        classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
        )
    )


def train_and_predict_split(data: pd.DataFrame, context_length: int = 512):
    """
    Trains the Chronos2 model on the first 70% of the data and predicts the remaining 30%.

    Args:
        data (pd.DataFrame): The input DataFrame with a datetime index and 'close' column.
        context_length (int): The number of time steps to use as context for the prediction.
    """
    print("\n--- Training on 70% and Predicting 30% ---")

    # === 1. Prepare data for Chronos2 format ===
    data_chronos = data.copy()
    data_chronos = data_chronos.reset_index()
    data_chronos.rename(
        columns={"timestamp": "timestamp", "Close": "target"}, inplace=True
    )
    data_chronos["id"] = "BTC/USDT"

    # === 2. Split data ===
    split_index = int(len(data_chronos) * 0.7)
    # Use the last `context_length` points of the training data as context
    context_df = data_chronos.iloc[max(0, split_index - context_length) : split_index]
    actual_df = data_chronos.iloc[split_index:]
    prediction_length = len(actual_df)

    print(f"Training context size: {len(context_df)}")
    print(f"Prediction length: {prediction_length}")

    if len(context_df) < 2:
        print("Error: Not enough data in the context window to make a prediction.")
        return

    # === 3. Generate predictions with a rolling window (expanding context) ===
    print("Generating predictions for the test set using a rolling window...")

    predictions_list = []
    current_context_df = context_df.copy()

    for i in trange(len(actual_df), desc="Rolling Window Prediction"):
        if len(current_context_df) < 2:
            print(f"Skipping step {i + 1} due to insufficient context.")
            continue

        try:
            # Predict one step ahead
            single_pred_df = pipeline.predict_df(
                current_context_df,
                prediction_length=1,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
            predictions_list.append(single_pred_df)

            # Update context for the next step with the true value from actual_df (expanding window)
            next_actual_step = actual_df.iloc[[i]]
            current_context_df = pd.concat(
                [current_context_df, next_actual_step], ignore_index=True
            )

        except Exception as e:
            print(
                f"Prediction failed at step {i + 1} with error: {e}. Stopping prediction."
            )
            break  # Stop if one prediction fails

    if not predictions_list:
        print("No predictions were made.")
        return

    # Concatenate all single-step predictions into one DataFrame
    pred_df = pd.concat(predictions_list, ignore_index=True)

    # === 4. Evaluate and Plot ===
    # Merge predictions with actual values
    results_df = pd.merge(
        actual_df[["timestamp", "target"]], pred_df, on="timestamp", how="inner"
    )
    if results_df.empty:
        print("Could not merge predictions with actuals. Check timestamp alignment.")
        return

    # Calculate RMSE on the median forecast
    rmse = np.sqrt(mean_squared_error(results_df["target"], results_df["0.5"]))
    print(f"\nEvaluation RMSE (on median forecast): {rmse:.4f}")

    # Plotting
    plt.figure(figsize=(15, 8))
    plt.plot(
        data_chronos["timestamp"],
        data_chronos["target"],
        label="Full Time Series",
        color="gray",
        alpha=0.6,
    )
    plt.plot(
        results_df["timestamp"],
        results_df["target"],
        label="Actual Values (Test Set)",
        color="blue",
    )
    plt.plot(
        results_df["timestamp"],
        results_df["0.5"],
        label="Predicted Median (0.5)",
        color="orange",
        linestyle="--",
    )

    # Plot uncertainty bounds
    plt.fill_between(
        results_df["timestamp"],
        results_df["0.1"],
        results_df["0.9"],
        color="orange",
        alpha=0.2,
        label="Uncertainty (10th-90th percentile)",
    )

    plt.title("Chronos2: 70/30 Train/Test Split Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to load data and run the Optuna study.
    """
    print("Loading historical data...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
    )

    # run_study(data, STUDY_NAME, N_TRIALS)
    # train_and_predict_split(data)
    train_and_predict_direction_split(data)


if __name__ == "__main__":
    main()
