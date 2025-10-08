import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr
import optuna
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import cupy as cp

from backtest_utils import fetch_historical_data


# --- Part 1: Data Simulation and Volume Bar Creation ---
# The paper uses high-frequency data to construct volume bars.
# We'll simulate 1-minute data to demonstrate the process.


def aggregate_to_volume_bars(df, volume_threshold=0):
    """
    Aggregates time-series data into volume bars based on a volume threshold.
    This follows the core concept of the dissertation (Section 3.3).
    """
    print(f"Step 2: Aggregating data into volume bars of {volume_threshold} units...")
    bars = []
    current_bar_data = []
    cumulative_volume = 0

    for index, row in df.iterrows():
        current_bar_data.append(row)
        # AI the volume value is in BTC. Multiply volume by the btc price  AI!
        cumulative_volume += row["volume"]
        print(cumulative_volume)
        if cumulative_volume >= volume_threshold:
            bar_df = pd.DataFrame(current_bar_data)

            # Bar characteristics
            open_time = bar_df.index[0]
            close_time = bar_df.index[-1]
            open_price = bar_df["close"].iloc[0]
            close_price = bar_df["close"].iloc[-1]

            # Calculate intra-bar volatility for labeling (σv)
            # The paper uses log-returns for some calculations.
            bar_log_returns = np.log(
                bar_df["close"] / bar_df["close"].shift(1)
            ).dropna()
            intra_bar_std = bar_log_returns.std()

            bars.append(
                {
                    "open_time": open_time,
                    "close_time": close_time,
                    "open_price": open_price,
                    "close_price": close_price,
                    "total_volume": cumulative_volume,
                    "intra_bar_std": intra_bar_std if len(bar_log_returns) > 1 else 0,
                }
            )

            # Reset for the next bar
            current_bar_data = []
            cumulative_volume = 0

    volume_bars_df = pd.DataFrame(bars)

    if volume_bars_df.empty:
        print("Warning: No volume bars created. Threshold might be too high for the dataset.")
        return volume_bars_df

    # Calculate bar returns (rv), which will be used for labeling
    volume_bars_df["bar_return"] = (
        volume_bars_df["close_price"] / volume_bars_df["open_price"]
    ) - 1

    print(f"Aggregation complete. {len(volume_bars_df)} volume bars created.\n")
    return volume_bars_df


# --- Part 2: Target Labeling and Feature Engineering ---


def create_labels(df, tau=0.35):
    """
    Creates target labels based on the triple-barrier method variation
    described in Section 3.3.2, Equation 3.1.
    """
    print("Step 3: Creating target labels...")
    df["label"] = 0
    # Shift returns and volatility to use future information for labeling ONLY
    df["next_bar_return"] = df["bar_return"].shift(-1)

    # Conditions for label = 1 ('top')
    # Condition 1: Next bar's return must be positive.
    cond1 = df["next_bar_return"] >= 0
    # Condition 2: Next bar's return must exceed current return + volatility threshold.
    cond2 = df["next_bar_return"] >= (df["bar_return"] + df["intra_bar_std"] * tau)

    df.loc[cond1 & cond2, "label"] = 1

    # Clean up columns used only for labeling
    df.dropna(subset=["next_bar_return"], inplace=True)
    df.drop(columns=["next_bar_return"], inplace=True)

    print(
        f"Labeling complete. Class distribution:\n{df['label'].value_counts(normalize=True)}\n"
    )
    return df


def create_features(df):
    """
    Creates simple features for the model. The paper uses 796 features;
    we will simulate a few for demonstration (Section 3.3.3).
    """
    print("Step 4: Creating features...")
    df["feature_return_lag_1"] = df["bar_return"].shift(1)
    df["feature_volatility_lag_1"] = df["intra_bar_std"].shift(1)
    df["feature_rolling_mean_return_5"] = (
        df["bar_return"].shift(1).rolling(window=5).mean()
    )
    df["feature_rolling_std_return_5"] = (
        df["bar_return"].shift(1).rolling(window=5).std()
    )

    # Drop rows with NaNs created by lagging/rolling features
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("Features created.\n")
    return df


def select_features(X: pd.DataFrame, y: pd.Series, corr_threshold=0.7, p_value_threshold=0.1) -> list:
    """
    Selects features based on Pearson correlation and p-value.
    """
    selected_features = []
    for col in X.columns:
        # Drop rows with NaN in either column for correlation calculation
        temp_df = pd.concat([X[col], y], axis=1).dropna()
        if len(temp_df) < 2:
            continue

        corr, p_value = pearsonr(temp_df.iloc[:, 0], temp_df.iloc[:, 1])
        if abs(corr) >= corr_threshold and p_value < p_value_threshold:
            selected_features.append(col)

    print(f"Selected {len(selected_features)} features out of {len(X.columns)} based on correlation criteria.")
    return selected_features


# --- Part 3: Model Training and Evaluation ---


def manual_backtest(X: pd.DataFrame, y: pd.Series, model, test_size: float = 0.3, refit_every: int = 1):
    """
    Performs a manual walk-forward backtest with an expanding window.
    The model is refit every `refit_every` steps.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        model: A scikit-learn compatible classifier.
        test_size (float): The proportion of the dataset to be used for testing.
        refit_every (int): How often (in steps) to refit the model. Default is 1 (every step).
    """
    split_index = int(len(X) * (1 - test_size))
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    y_pred = []
    scaler = StandardScaler()

    print("Starting walk-forward backtest...")
    # Walk forward
    for i in range(len(X_test)):
        current_split_index = split_index + i

        # Periodically refit the model on an expanding window
        if i % refit_every == 0:
            print(f"Refitting model at step {i+1}/{len(X_test)}...")
            X_train_current = X.iloc[:current_split_index]
            y_train_current = y.iloc[:current_split_index]

            # Scale training data
            X_train_current_scaled = scaler.fit_transform(X_train_current)

            # Calculate sample weights for balanced classes
            classes = np.unique(y_train_current)
            sample_weights_gpu = None
            if len(classes) > 1:
                weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_current)
                class_weight_dict = dict(zip(classes, weights))
                sample_weights = y_train_current.map(class_weight_dict).to_numpy()
                sample_weights_gpu = cp.asarray(sample_weights)


            # Move data to GPU and retrain model
            X_train_gpu = cp.asarray(X_train_current_scaled)
            y_train_gpu = cp.asarray(y_train_current)
            
            model.fit(X_train_gpu, y_train_gpu, sample_weight=sample_weights_gpu)

        # Get current test sample and scale it using the latest scaler
        X_test_current = X.iloc[current_split_index:current_split_index + 1]
        X_test_current_scaled = scaler.transform(X_test_current)

        # Move data to GPU for prediction
        X_test_gpu = cp.asarray(X_test_current_scaled)

        # Predict
        prediction = model.predict(X_test_gpu)
        y_pred.append(int(cp.asnumpy(prediction)[0]))

        if (i + 1) % 50 == 0 or (i + 1) == len(X_test):
            print(f"Processed {i+1}/{len(X_test)} steps...")

    # Evaluate
    print("\n--- Backtest Classification Report ---")
    target_names = ['non-top (0)', 'top (1)']
    labels = [0, 1]
    report_str = classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0)
    print(report_str)
    report_dict = classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0, output_dict=True)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Backtest Accuracy: {accuracy:.2f}")

    return y_pred, y_test, report_dict


def objective(trial: optuna.Trial, minute_data: pd.DataFrame) -> float:
    """
    Optuna objective function to tune hyperparameters for the Palazzo price reversal model.
    """
    # === 1. Define Hyperparameter Search Space ===
    # Data aggregation and labeling hyperparameters
    volume_threshold = trial.suggest_int('volume_threshold', 0, 0)
    tau = trial.suggest_float('tau', 0.1, 0.6)

    # Feature selection hyperparameter
    corr_threshold = trial.suggest_float('corr_threshold', 0.01, 0.5)
    p_value_threshold = trial.suggest_float('p_value_threshold', 0.01, 0.2)


    # Model hyperparameters for XGBoost
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        "seed": 42,
    }
    refit_every = 24

    # === 2. Run the ML Pipeline ===
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: {trial.params}")

    # Aggregate into volume bars
    volume_bars = aggregate_to_volume_bars(minute_data, volume_threshold=volume_threshold)

    if volume_bars.empty:
        print("No volume bars created. Pruning trial.")
        raise optuna.exceptions.TrialPruned()
    
    # Create target labels
    labeled_bars = create_labels(volume_bars.copy(), tau=tau)
    
    # Engineer features
    final_df = create_features(labeled_bars)
    
    if final_df.empty:
        print("No data after feature engineering. Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    y = final_df['label']
    features = [col for col in final_df.columns if "feature_" in col]
    X = final_df[features]

    # Prune trial if not enough positive samples are found
    if y.sum() < 10:
        print("Not enough positive samples found with these parameters. Pruning trial.")
        raise optuna.exceptions.TrialPruned()
        
    # Feature Selection
    selected_cols = select_features(X, y, corr_threshold=corr_threshold, p_value_threshold=p_value_threshold)
    if not selected_cols:
        print("No features selected. Pruning trial.")
        raise optuna.exceptions.TrialPruned()
        
    X = X[selected_cols]
    
    # Run Backtest
    model = xgb.XGBClassifier(**params)
    _, _, report = manual_backtest(X, y, model, test_size=0.3, refit_every=refit_every)

    # === 3. Calculate and Return the Objective Metric ===
    f1_top = report.get('top (1)', {}).get('f1-score', 0.0)
    
    print(f"Trial {trial.number} finished. F1 (Top): {f1_top:.4f}")
    
    return f1_top


def main():
    """
    Main function to run the Optuna hyperparameter optimization study.
    """
    N_TRIALS = 1

    # 1. Load high-frequency data
    print("Loading 1-minute historical data...")
    minute_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        start_date="2024-01-01T00:00:00Z"
    )
    # The aggregate_to_volume_bars function expects 'close' and 'volume' columns.
    # fetch_historical_data returns 'Close' and 'Volume', so we rename them.
    minute_data.rename(columns={'Close': 'close', 'Volume': 'volume'}, inplace=True)

    # 2. Setup and Run Optuna Study
    db_file_name = "optuna-study"
    study_name_in_db = 'xgboost_price_reversal_palazzo_v1'
    storage_name = f"sqlite:///{db_file_name}.db"

    print(f"Starting Optuna study: '{study_name_in_db}'. Storage: {storage_name}")

    # Use a partial function to pass the loaded data to the objective function
    objective_with_data = partial(objective, minute_data=minute_data)

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name_in_db,
        storage=storage_name,
        load_if_exists=True
    )

    study.optimize(objective_with_data, n_trials=N_TRIALS)
    
    # 3. Print Study Results
    print("\n--- Optuna Study Best Results ---")
    try:
        best_trial = study.best_trial
        print(f"Best trial value (F1 Score): {best_trial.value}")
        print("Best parameters found:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    except ValueError:
        print("No successful trials were completed.")


if __name__ == "__main__":
    main()
