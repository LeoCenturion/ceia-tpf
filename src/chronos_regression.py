import pandas as pd
import numpy as np
from autogluon.multimodal import MultiModalPredictor
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import optuna
from functools import partial
import matplotlib.pyplot as plt

from backtest_utils import fetch_historical_data


def run_multimodal_regression_backtest():
    """
    Performs a walk-forward backtest for a regression task predicting the next closing price.
    Uses 'Open', 'Close', 'High', 'Low' as features. Saves predictions, actuals,
    and periods since last refit to disk.
    """
    print("\n--- Running Regression Walk-Forward Backtest to Predict Closing Price ---")

    # 1. Load Data
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        timeframe="1h",
        start_date="2022-01-01T00:00:00Z",
    )

    # 2. Prepare Features and Target
    print("Preparing features (Open, Close, High, Low) and target (next Close)...")
    features_df = data[["Open", "High", "Low", "Close"]].copy()
    target = data["Close"].shift(-1)
    target.name = "target_close_price"

    final_df = pd.concat([features_df, target], axis=1).dropna()
    X = final_df.drop(columns=["target_close_price"])
    y = final_df["target_close_price"]

    # 3. Define Backtesting Parameters
    test_size = 0.3
    refit_every = 24 * 365 * 1  # Refit every week (hourly data)

    # FT-Transformer Hyperparameters (example values, could be optimized with Optuna)
    hyperparameters = {
        "model.names": ["ft_transformer"],
        "model.ft_transformer.num_blocks": 4,
        "model.ft_transformer.hidden_size": 192,
        "model.ft_transformer.token_dim": 192,
        "model.ft_transformer.ffn_hidden_size": 192 * 2,
        "optim.lr": 0.0005,
        "optim.weight_decay": 1e-4,
        "env.per_gpu_batch_size": 128,
    }

    # 4. Perform Walk-Forward Backtest
    split_index = int(len(final_df) * (1 - test_size))

    predictions_df_list = []
    predictor = None
    periods_since_refit = 0

    print("Starting walk-forward backtest...")
    for i in range(split_index, len(final_df)):
        current_data_index = i

        # Periodically refit the model on an expanding window
        if periods_since_refit % refit_every == 0:
            print(
                f"Refitting model at step {i - split_index + 1}/{len(final_df) - split_index} (index {current_data_index})..."
            )
            train_X = X.iloc[:current_data_index]
            train_y = y.iloc[:current_data_index]

            predictor = MultiModalPredictor(
                label="target_close_price", problem_type="regression", eval_metric="r2"
            )
            predictor.set_verbosity(0)  # errors
            predictor.fit(
                pd.concat(
                    [train_X, train_y], axis=1
                ),  # MultiModalPredictor expects df with target
                hyperparameters=hyperparameters,
                time_limit=300,  # Shorter time limit for refitting steps
            )
            periods_since_refit = 0

        # Make prediction for the current step
        if predictor:
            current_X_test = X.iloc[current_data_index : current_data_index + 1]
            predicted_price = predictor.predict(current_X_test, as_pandas=False)
        else:
            predicted_price = np.nan  # No prediction if model not yet fitted

        actual_price = y.iloc[current_data_index]

        predictions_df_list.append(
            {
                "timestamp": y.index[current_data_index],
                "actual_close": actual_price,
                "predicted_close": predicted_price,
                "periods_since_refit": periods_since_refit,
            }
        )

        periods_since_refit += 1

        if (i - split_index + 1) % 100 == 0 or (i == len(final_df) - 1):
            print(
                f"Processed {i - split_index + 1}/{len(final_df) - split_index} steps..."
            )

    results_df = pd.DataFrame(predictions_df_list).set_index("timestamp")

    # 5. Save results to disk
    output_path = "regression_backtest_results.csv"
    results_df.to_csv(output_path)
    print(f"\nBacktest results saved to {output_path}")

    # 6. Evaluate overall performance and plot
    final_y_true = results_df["actual_close"].dropna()
    final_y_pred = results_df["predicted_close"].dropna()

    if not final_y_true.empty and not final_y_pred.empty:
        # Align indices before evaluation
        common_index = final_y_true.index.intersection(final_y_pred.index)
        final_y_true = final_y_true.loc[common_index]
        final_y_pred = final_y_pred.loc[common_index]

        from sklearn.metrics import mean_squared_error, r2_score

        rmse = np.sqrt(mean_squared_error(final_y_true, final_y_pred))
        r2 = r2_score(final_y_true, final_y_pred)
        print(f"Overall Backtest RMSE: {rmse:.4f}")
        print(f"Overall Backtest R^2: {r2:.4f}")

        plt.figure(figsize=(15, 7))
        plt.plot(
            final_y_true.index, final_y_true, label="Actual Close Price", color="blue"
        )
        plt.plot(
            final_y_pred.index,
            final_y_pred,
            label="Predicted Close Price",
            color="red",
            linestyle="--",
        )
        plt.title("Closing Price Regression Backtest: Actual vs. Predicted")
        plt.xlabel("Date")
        plt.ylabel("Price (USDT)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        print("\nDisplaying plot. Close the plot window to exit.")
        plt.show()
    else:
        print("\nNo valid predictions were made for overall evaluation.")


def run_timeseries_regression_backtest():
    """
    Performs a walk-forward backtest for a regression task predicting the next closing price.
    Uses 'Open', 'Close', 'High', 'Low' as features. Saves predictions, actuals,
    and periods since last refit to disk.
    """
    print("\n--- Running Regression Walk-Forward Backtest to Predict Closing Price ---")

    # 1. Load Data
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        timeframe="1h",
        start_date="2022-01-01T00:00:00Z",
    )

    # 2. Prepare Features and Target
    print("Preparing features (Open, Close, High, Low) and target (next Close)...")
    features_df = data[["Open", "High", "Low", "Close"]].copy()
    target = data["Close"].shift(-1)
    target.name = "target"

    final_df = pd.concat([features_df, target], axis=1).dropna()
    X = final_df.drop(columns=["target"])
    X["item_id"] = "series_0"
    X["timestamp"] = X.index.values
    y = final_df["target"]

    # 3. Define Backtesting Parameters
    test_size = 0.3
    refit_every = 24 * 14  # Refit every week (hourly data)

    # 4. Perform Walk-Forward Backtest
    split_index = int(len(final_df) * (1 - test_size))

    predictions_df_list = []
    predictor = None
    periods_since_refit = 0

    print("Starting walk-forward backtest...")
    for i in range(split_index, len(final_df)):
        current_data_index = i

        # Periodically refit the model on an expanding window
        if periods_since_refit % refit_every == 0:
            print(
                f"Refitting model at step {i - split_index + 1}/{len(final_df) - split_index} (index {current_data_index})..."
            )
            train_X = X.iloc[:current_data_index]
            train_y = y.iloc[:current_data_index]

            # Combine target and features into a single DataFrame
            df = pd.concat([train_y, train_X], axis=1)

            # Convert to TimeSeriesDataFrame
            # static_features are not used here, but id_column and timestamp_column are essential
            train_data = TimeSeriesDataFrame.from_data_frame(
                df,
                id_column="item_id",
                timestamp_column="timestamp",  # The index is automatically named 'timestamp'
            )
            predictor = TimeSeriesPredictor(prediction_length=1, verbosity=0, freq="1h")
            predictor.fit(
                train_data,
                hyperparameters={
                    "Chronos": {"model_path": "amazon/chronos-bolt-base"},
                },
            )
            periods_since_refit = 0

        # Make prediction for the current step
        if predictor:
            train_X = X.iloc[:current_data_index]
            train_y = y.iloc[:current_data_index]
            df = pd.concat([train_y, train_X], axis=1)
            predicted_price = predictor.predict(df)["mean"].values[0]
            print(predicted_price)

        else:
            predicted_price = np.nan  # No prediction if model not yet fitted

        actual_price = y.iloc[current_data_index]

        predictions_df_list.append(
            {
                "timestamp": y.index[current_data_index],
                "actual_close": actual_price,
                "predicted_close": predicted_price,
                "periods_since_refit": periods_since_refit,
            }
        )

        periods_since_refit += 1

        if (i - split_index + 1) % 100 == 0 or (i == len(final_df) - 1):
            print(
                f"Processed {i - split_index + 1}/{len(final_df) - split_index} steps..."
            )

    results_df = pd.DataFrame(predictions_df_list).set_index("timestamp")

    # 5. Save results to disk
    output_path = "regression_backtest_results.csv"
    results_df.to_csv(output_path)
    print(f"\nBacktest results saved to {output_path}")

    # 6. Evaluate overall performance and plot
    final_y_true = results_df["actual_close"].dropna()
    final_y_pred = results_df["predicted_close"].dropna()

    if not final_y_true.empty and not final_y_pred.empty:
        # Align indices before evaluation
        common_index = final_y_true.index.intersection(final_y_pred.index)
        final_y_true = final_y_true.loc[common_index]
        final_y_pred = final_y_pred.loc[common_index]

        from sklearn.metrics import mean_squared_error, r2_score

        rmse = np.sqrt(mean_squared_error(final_y_true, final_y_pred))
        r2 = r2_score(final_y_true, final_y_pred)
        print(f"Overall Backtest RMSE: {rmse:.4f}")
        print(f"Overall Backtest R^2: {r2:.4f}")

        plt.figure(figsize=(15, 7))
        plt.plot(
            final_y_true.index, final_y_true, label="Actual Close Price", color="blue"
        )
        plt.plot(
            final_y_pred.index,
            final_y_pred,
            label="Predicted Close Price",
            color="red",
            linestyle="--",
        )
        plt.title("Closing Price Regression Backtest: Actual vs. Predicted")
        plt.xlabel("Date")
        plt.ylabel("Price (USDT)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        print("\nDisplaying plot. Close the plot window to exit.")
        plt.show()
    else:
        print("\nNo valid predictions were made for overall evaluation.")


def objective(trial, data):
    """
    Optuna objective function for time-series regression backtest.
    """
    from sklearn.metrics import mean_squared_error

    model_choice = trial.suggest_categorical(
        "chronos_model",
        ["amazon/chronos-t5-tiny", "amazon/chronos-t5-small", "amazon/chronos-t5-base"],
    )
    refit_every = trial.suggest_int("refit_every_hours", 24 * 7, 24 * 14)

    # 2. Prepare Features and Target
    features_df = data[["Open", "High", "Low", "Close"]].copy()
    target = data["Close"].shift(-1)
    target.name = "target"

    final_df = pd.concat([features_df, target], axis=1).dropna()
    X = final_df.drop(columns=["target"])
    X["item_id"] = "series_0"
    X["timestamp"] = X.index.values
    y = final_df["target"]

    # Combine into a single DataFrame for AutoGluon to avoid repeated concatenation
    autogluon_df = pd.concat([y, X], axis=1)

    # 3. Define Backtesting Parameters (use a smaller test set for faster trials)
    test_size = 0.1
    split_index = int(len(final_df) * (1 - test_size))

    predictions_df_list = []
    predictor = None
    periods_since_refit = 0

    # 4. Perform Walk-Forward Backtest for the trial
    print(
        f"Trial {trial.number}: Starting backtest with model {model_choice} and refit_every={refit_every} hours."
    )
    for i in range(split_index, len(final_df)):
        current_data_index = i

        if periods_since_refit % refit_every == 0:
            print(
                f"Trial {trial.number}: Refitting model at step {i - split_index + 1}/{len(final_df) - split_index}..."
            )
            train_df = autogluon_df.iloc[:current_data_index]
            train_data = TimeSeriesDataFrame.from_data_frame(
                train_df, id_column="item_id", timestamp_column="timestamp"
            )
            predictor = TimeSeriesPredictor(prediction_length=1, verbosity=0, freq="1h")
            try:
                predictor.fit(
                    train_data,
                    hyperparameters={
                        "Chronos": {
                            "model_path": model_choice,
                            "fine_tune_batch_size": 1024,
                            "batch_size": 1024,
                        }
                    },
                    time_limit=300,
                )
            except Exception as e:
                print(f"Trial {trial.number} failed during fit: {e}")
                return np.inf  # Return high value if model fails to fit
            periods_since_refit = 0

        actual_price = y.iloc[current_data_index]
        log_entry = {
            "timestamp": y.index[current_data_index],
            "actual_close": actual_price,
            "hours_since_last_refit": periods_since_refit,
        }

        if predictor:
            predict_context_df = autogluon_df.iloc[:current_data_index]
            prediction_df = predictor.predict(predict_context_df)
            if not prediction_df.empty:
                prediction_values = prediction_df.iloc[0].to_dict()
                log_entry.update(prediction_values)

        predictions_df_list.append(log_entry)

        periods_since_refit += 1

    results_df = pd.DataFrame(predictions_df_list).set_index("timestamp")

    # Save predictions to a CSV for each trial
    predictions_filename = f"predictions_trial_{trial.number}.csv"
    results_df.to_csv(predictions_filename)

    # 5. Calculate and return RMSE
    final_y_true = results_df["actual_close"].dropna()
    final_y_pred = results_df["mean"].dropna()

    if final_y_true.empty or final_y_pred.empty:
        print(f"Trial {trial.number}: No valid predictions generated.")
        return np.inf

    common_index = final_y_true.index.intersection(final_y_pred.index)
    final_y_true = final_y_true.loc[common_index]
    final_y_pred = final_y_pred.loc[common_index]

    if len(final_y_true) == 0:
        print(f"Trial {trial.number}: No overlapping predictions and actuals.")
        return np.inf

    rmse = np.sqrt(mean_squared_error(final_y_true, final_y_pred))
    print(f"Trial {trial.number}: Finished with RMSE: {rmse}")
    return rmse


def run_study(data, study_name_in_db, ntrials):
    # 2. Setup and Run Optuna Study
    db_file_name = "optuna-study"

    storage_name = f"sqlite:///{db_file_name}.db"

    print(f"Starting Optuna study: '{study_name_in_db}'. Storage: {storage_name}")

    objective_with_data = partial(objective, data=data)

    study = optuna.create_study(
        direction="minimize",  # We want to minimize RMSE
        study_name=study_name_in_db,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(
        objective_with_data, n_trials=ntrials, n_jobs=1
    )  # n_jobs=1 for GPU usage

    # 3. Print Study Results
    print("\n--- Optuna Study Best Results ---")
    try:
        best_trial = study.best_trial
        print(f"Best trial value (RMSE): {best_trial.value}")
        print("Best parameters found:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    except ValueError:
        print("No successful trials were completed.")


def main():
    """
    Main function to run the Optuna hyperparameter optimization study.
    """

    # 1. Load Data
    print("Loading historical data...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        timeframe="1h",
        start_date="2025-01-01T00:00:00Z",
    )
    # Precompute percentage change for OHLC data
    # for col in ["Open", "Close", "High", "Low"]:
    #     data[col] = data[col].pct_change()
    # data.dropna(inplace=True)
    study_name_in_db = "chronos_pct_change_regression_v0.2"
    run_study(data, study_name_in_db, 1)


if __name__ == "__main__":
    main()  # Uncomment to run the Optuna study
    # run_single_backtest()
    # run_regression_evaluation() # This was the single train/test evaluation
    # run_timeseries_regression_backtest() # This is the new walk-forward backtest
