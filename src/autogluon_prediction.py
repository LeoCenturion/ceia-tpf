from functools import partial

import optuna
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score

from backtest_utils import fetch_historical_data
from xgboost_price_reversal import create_features, create_target_variable


def objective(trial, data):
    """Optuna objective function for tabular classification backtest."""
    # Hyperparameters to tune
    target_method = trial.suggest_categorical(
        "target_method", ["pct_change_std", "ao_on_pct_change", "ao_on_price"]
    )
    refit_every = trial.suggest_int("refit_every_hours", 24 * 7, 24 * 30, step=24)
    presets = trial.suggest_categorical(
        "presets", ["medium_quality", "high_quality", "best_quality"]
    )

    # 1. Create features and target variable
    print("Creating features and target variable...")
    features_df = create_features(data)
    data_with_target = create_target_variable(data.copy(), method=target_method)

    # We want to predict the next period's movement.
    # So, we use features from time 't' to predict the target at time 't+1'.
    target = data_with_target["target"].shift(-1)

    final_df = pd.concat([features_df, target], axis=1).dropna()
    final_df["target"] = final_df["target"].astype(int)

    X = final_df.drop(columns=["target"])
    y = final_df["target"]
    autogluon_df = pd.concat([X, y], axis=1)

    # 2. Backtesting parameters
    test_size = 0.3
    split_index = int(len(final_df) * (1 - test_size))

    predictions_list = []
    actuals_list = []
    predictor = None
    periods_since_refit = 0

    # 3. Walk-forward backtest loop
    print(
        f"Trial {trial.number}: Starting backtest with method='{target_method}', "
        f"presets='{presets}', refit_every={refit_every} hours."
    )

    for i in range(split_index, len(final_df)):
        current_data_index = i

        # Periodically refit the model on an expanding window
        if periods_since_refit % refit_every == 0:
            print(
                f"Trial {trial.number}: Refitting model at step "
                f"{i - split_index + 1}/{len(final_df) - split_index}..."
            )
            train_df = autogluon_df.iloc[:current_data_index]

            predictor = TabularPredictor(
                label="target", problem_type="multiclass", eval_metric="f1_macro"
            )
            try:
                predictor.fit(train_df, presets=presets, time_limit=300, verbosity=0)
            except Exception as e:
                print(f"Trial {trial.number} failed during fit: {e}")
                return 1.0  # Return high error if model fails to fit

            periods_since_refit = 0

        if predictor:
            current_X_test = X.iloc[current_data_index : current_data_index + 1]
            try:
                predicted_label = predictor.predict(current_X_test).iloc[0]
                predictions_list.append(predicted_label)
                actuals_list.append(y.iloc[current_data_index])
            except Exception as e:
                print(f"Trial {trial.number} failed during prediction: {e}")
                # Skip this prediction but continue the backtest
                pass

        periods_since_refit += 1

    # 4. Evaluate and return the score
    if not predictions_list:
        print(f"Trial {trial.number}: No predictions were made.")
        return 1.0  # Return high error

    # Use F1 macro score for multiclass classification performance
    score = f1_score(actuals_list, predictions_list, average="macro", zero_division=0)
    print(f"Trial {trial.number}: Finished with F1 Macro score: {score:.4f}")

    # Optuna minimizes, so we return 1.0 - F1 score
    return 1.0 - score


def run_study(data, study_name_in_db, ntrials):
    """Sets up and runs an Optuna study for the given data."""
    db_file_name = "optuna-study-classification"
    storage_name = f"sqlite:///{db_file_name}.db"

    print(f"Starting Optuna study: '{study_name_in_db}'. Storage: {storage_name}")

    objective_with_data = partial(objective, data=data)

    study = optuna.create_study(
        direction="minimize",  # We want to minimize (1 - F1 score)
        study_name=study_name_in_db,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(objective_with_data, n_trials=ntrials, n_jobs=1)

    print("\n--- Optuna Study Best Results ---")
    try:
        best_trial = study.best_trial
        print(f"Best trial value (1 - F1 Macro): {best_trial.value:.4f}")
        print("Best parameters found:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    except ValueError:
        print("No successful trials were completed.")


def main():
    """Main function to load data and run the Optuna study for classification."""
    print("Loading historical data for AutoGluon classification task...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        timeframe="1h",
        start_date="2022-01-01T00:00:00Z",
    )
    study_name_in_db = "autogluon_tabular_classification_v1"
    run_study(data, study_name_in_db, ntrials=50)


if __name__ == "__main__":
    main()
