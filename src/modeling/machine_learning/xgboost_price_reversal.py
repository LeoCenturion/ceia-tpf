from functools import partial

import cupy as cp
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.data_analysis import fetch_historical_data
from src.data_analysis.indicators import (
    create_ao_target,
    create_features,
)


def select_features(
    X: pd.DataFrame, y: pd.Series, corr_threshold=0.7, p_value_threshold=0.1
) -> list:
    """
    Selects features based on Pearson correlation and p-value.
    Note: The correlation threshold of 0.7 is extremely high and might result in
    very few or no features being selected. The paper's methodology is followed here.
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

    print(
        f"Selected {len(selected_features)} features out of {len(X.columns)} based on correlation criteria."
    )
    return selected_features


def manual_backtest(
    X: pd.DataFrame, y: pd.Series, model, test_size: float = 0.3, refit_every: int = 1
):
    """
    Performs a manual walk-forward backtest with an expanding window.
    The model is refit every `refit_every` steps.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector (must contain labels 0, 1, 2).
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
            print(f"Refitting model at step {i + 1}/{len(X_test)}...")
            X_train_current = X.iloc[:current_split_index]
            y_train_current = y.iloc[:current_split_index]

            # Scale training data
            X_train_current_scaled = scaler.fit_transform(X_train_current)

            # Calculate sample weights for balanced classes
            classes = np.unique(y_train_current)
            weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=y_train_current
            )
            class_weight_dict = dict(zip(classes, weights))
            sample_weights = y_train_current.map(class_weight_dict).to_numpy()

            # Move data to GPU and retrain model
            X_train_gpu = cp.asarray(X_train_current_scaled)
            y_train_gpu = cp.asarray(y_train_current)
            sample_weights_gpu = cp.asarray(sample_weights)
            model.fit(X_train_gpu, y_train_gpu, sample_weight=sample_weights_gpu)

        # Get current test sample and scale it using the latest scaler
        X_test_current = X.iloc[current_split_index : current_split_index + 1]
        X_test_current_scaled = scaler.transform(X_test_current)

        # Move data to GPU for prediction
        X_test_gpu = cp.asarray(X_test_current_scaled)

        # Predict
        prediction = model.predict(X_test_gpu)
        y_pred.append(int(cp.asnumpy(prediction)[0]))

        if (i + 1) % 50 == 0 or (i + 1) == len(X_test):
            print(f"Processed {i + 1}/{len(X_test)} steps...")

    # Evaluate
    print("\n--- Backtest Classification Report ---")
    target_names = ["Bottom (-1)", "Neutral (0)", "Top (1)"]
    labels = [0, 1, 2]  # The labels in y_test and y_pred
    report_str = classification_report(
        y_test, y_pred, labels=labels, target_names=target_names, zero_division=0
    )
    print(report_str)
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Backtest Accuracy: {accuracy:.2f}")

    return y_pred, y_test, report_dict


def plot_reversals_on_candlestick(
    data: pd.DataFrame, reversal_points: pd.DataFrame, sample_size: int = None
):
    """
    Plots a candlestick chart with markers for identified tops and bottoms.

    Args:
        data (pd.DataFrame): The original DataFrame with 'Open', 'High', 'Low', 'Close' prices.
        reversal_points (pd.DataFrame): DataFrame containing the identified tops (target=1) and bottoms (target=-1).
        sample_size (int, optional): The number of recent data points to plot. If None, plots the entire series.
    """
    plot_data = data.copy()
    plot_reversal_points = reversal_points

    if sample_size:
        plot_data = plot_data.tail(sample_size)
        # Filter reversal points to be within the plotted data's index range
        plot_reversal_points = reversal_points[
            reversal_points.index >= plot_data.index[0]
        ]

    # Create series for plotting markers
    tops = plot_reversal_points[plot_reversal_points["target"] == 1]
    bottoms = plot_reversal_points[plot_reversal_points["target"] == -1]

    # Place markers slightly above highs for tops and below lows for bottoms
    top_markers = pd.Series(np.nan, index=plot_data.index)
    bottom_markers = pd.Series(np.nan, index=plot_data.index)

    top_indices = tops.index.intersection(plot_data.index)
    bottom_indices = bottoms.index.intersection(plot_data.index)

    # Check if there are any tops/bottoms to plot to avoid errors on empty access
    if not top_indices.empty:
        top_markers.loc[top_indices] = plot_data.loc[top_indices, "High"] * 1.01
    if not bottom_indices.empty:
        bottom_markers.loc[bottom_indices] = plot_data.loc[bottom_indices, "Low"] * 0.99

    # Create addplots for mplfinance
    addplots = [
        mpf.make_addplot(
            top_markers, type="scatter", marker="^", color="green", markersize=100
        ),
        mpf.make_addplot(
            bottom_markers, type="scatter", marker="v", color="red", markersize=100
        ),
    ]

    title = "Candlestick Chart with Tops and Bottoms"
    if sample_size:
        title += f" (Last {sample_size} hours)"

    mpf.plot(
        plot_data,
        type="candle",
        style="yahoo",
        title=title,
        ylabel="Price ($)",
        addplot=addplots,
        figsize=(15, 7),
        volume=True,
        panel_ratios=(3, 1),
    )


def objective(trial: optuna.Trial, data: pd.DataFrame) -> float:
    """
    Optuna objective function to tune hyperparameters for the XGBoost price reversal model.
    """
    # === 1. Define Hyperparameter Search Space ===
    # Peak detection hyperparameters
    peak_method = trial.suggest_categorical("peak_method", ["pct_change_on_ao"])

    std_fraction = 1.0
    if peak_method == "pct_change_std":
        std_fraction = trial.suggest_float("std_fraction", 0.5, 2.0)
        # These are not used for 'pct_change_std' but need to be defined
        peak_distance = 1
        peak_threshold = 0
    else:
        peak_distance = trial.suggest_int("peak_distance", 1, 5)
        if peak_method == "ao_on_price":
            # Threshold is a difference, so values can be smaller than absolute price
            peak_threshold = trial.suggest_float("peak_threshold", 0.0, 100.0)
        elif peak_method == "pct_change_on_ao":
            peak_threshold = trial.suggest_float("peak_threshold", 0.0, 5)
        else:  # For pct_change based methods, the scale is much smaller
            peak_threshold = trial.suggest_float("peak_threshold", 0.0, 0.005)

    # Feature selection hyperparameter
    corr_threshold = trial.suggest_float("corr_threshold", 0.1, 0.7)

    # Model hyperparameters for XGBoost
    params = {
        "objective": "multi:softmax",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",  # Use GPU
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "n_jobs": -1,
    }
    refit_every = 24 * 7

    # === 2. Run the ML Pipeline ===
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: {trial.params}")

    # Create Target Variable
    reversal_data = create_ao_target(
        data.copy(),
        method=peak_method,
        peak_distance=peak_distance,
        peak_threshold=peak_threshold,
        std_fraction=std_fraction,
    )

    # Prune trial if not enough reversal points are found
    if (
        reversal_data["target"].nunique() < 3
        or reversal_data["target"].value_counts().get(1, 0) < 5
        or reversal_data["target"].value_counts().get(-1, 0) < 5
    ):
        print("Not enough reversal points found with these parameters. Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    y = reversal_data["target"]

    # Prune trial if initial training set is not representative
    split_index = int(
        len(reversal_data) * (1 - 0.3)
    )  # Corresponds to test_size in manual_backtest
    if y.iloc[:split_index].nunique() < 3:
        print("Initial training set does not contain all 3 classes. Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    # Remap labels for XGBoost, which requires labels in [0, num_class-1]
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})

    # Create Features
    features_df = create_features(data)
    X = features_df.loc[reversal_data.index]
    X = X.loc[:, (X != X.iloc[0]).any()]  # Drop constant columns

    # Feature Selection (use original y for correlation)
    selected_cols = select_features(X, y, corr_threshold=corr_threshold)
    if selected_cols:
        X = X[selected_cols]
    # Run Backtest
    model = xgb.XGBClassifier(**params)
    _, _, report = manual_backtest(
        X, y_mapped, model, test_size=0.3, refit_every=refit_every
    )

    # === 3. Calculate and Return the Objective Metric ===
    f1_top = report.get("Top (1)", {}).get("f1-score", 0.0)
    f1_bottom = report.get("Bottom (-1)", {}).get("f1-score", 0.0)

    # We want to maximize the average F1 score for identifying tops and bottoms
    objective_value = (f1_top + f1_bottom) / 2
    print(f"Trial {trial.number} finished. Avg F1 (Top/Bottom): {objective_value:.4f}")

    return objective_value


def main():
    """
    Main function to run the Optuna hyperparameter optimization study.
    """
    N_TRIALS = 50

    # 1. Load Data
    print("Loading historical data...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
    )

    # 2. Setup and Run Optuna Study
    db_file_name = "optuna-study"
    study_name_in_db = "xgboost price std reversal v2"
    storage_name = f"sqlite:///{db_file_name}.db"

    print(f"Starting Optuna study: '{study_name_in_db}'. Storage: {storage_name}")

    # Use a partial function to pass the loaded data to the objective function
    objective_with_data = partial(objective, data=data)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name_in_db,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(objective_with_data, n_trials=N_TRIALS, n_jobs=-1)
    # 3. Print Study Results
    print("\n--- Optuna Study Best Results ---")
    try:
        best_trial = study.best_trial
        print(f"Best trial value (Average F1 Score): {best_trial.value}")
        print("Best parameters found:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    except ValueError:
        print("No successful trials were completed.")


def plot_feature_selection_by_threshold(X: pd.DataFrame, y: pd.Series):
    """
    Plots the number of selected features for different correlation thresholds
    and shows which features are selected at each threshold.
    """
    # 1. Calculate absolute Pearson correlations for all features
    correlations = {}
    print("\nCalculating feature correlations...")
    for col in X.columns:
        temp_df = pd.concat([X[col], y], axis=1).dropna()
        if len(temp_df) < 2:
            correlations[col] = 0
            continue
        corr, _ = pearsonr(temp_df.iloc[:, 0], temp_df.iloc[:, 1])
        correlations[col] = abs(corr)

    corr_df = pd.DataFrame.from_dict(
        correlations, orient="index", columns=["correlation"]
    ).sort_values("correlation", ascending=False)

    # --- Plot 1: Number of features vs. Threshold ---
    thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)
    num_features = []
    print("\nAnalyzing feature selection across different correlation thresholds...")
    for threshold in thresholds:
        selected_count = (corr_df["correlation"] >= threshold).sum()
        num_features.append(selected_count)
        print(f"Threshold: {threshold:.1f}, Features selected: {selected_count}")

    plt.figure(figsize=(12, 7))
    bar_positions = np.arange(len(thresholds))
    bars = plt.bar(bar_positions, num_features)
    plt.title("Number of Selected Features vs. Correlation Threshold")
    plt.xlabel("Pearson Correlation Threshold")
    plt.ylabel("Number of Features Selected")
    plt.xticks(bar_positions, [f"{t:.1f}" for t in thresholds])
    plt.bar_label(bars)
    plt.grid(axis="y", linestyle="--")
    plt.show()

    # --- Plot 2: Heatmap of feature selection ---
    # Filter to only show features that are selected by at least the lowest threshold
    features_to_plot = corr_df[corr_df["correlation"] >= thresholds[0]]

    if features_to_plot.empty:
        print(
            "No features met the lowest correlation threshold of 0.1. Skipping heatmap."
        )
        return

    selection_matrix = pd.DataFrame(index=features_to_plot.index)
    for t in thresholds:
        selection_matrix[f">={t:.1f}"] = (features_to_plot["correlation"] >= t).astype(
            int
        )

    plt.figure(figsize=(10, max(8, len(features_to_plot.index) * 0.3)))
    sns.heatmap(
        selection_matrix, annot=True, cbar=False, cmap="viridis", linewidths=0.5
    )
    plt.title("Feature Selection at Different Correlation Thresholds")
    plt.xlabel("Correlation Threshold")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def show_plot():
    # --- Code to generate X and y for plot_feature_selection_by_threshold ---
    print("Preparing data for feature selection analysis plot...")
    # 1. Load data
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
    )
    # 2. Create a representative target variable for correlation analysis
    # The exact method doesn't matter as much as having a target to correlate against.
    reversal_data = create_ao_target(data.copy(), method="ao_on_price")
    y = reversal_data["target"]

    # 3. Create features
    features_df = create_features(data)
    X = features_df.loc[reversal_data.index]
    X = X.loc[:, (X != X.iloc[0]).any()]  # Drop constant columns

    # 4. Call the plotting function
    plot_feature_selection_by_threshold(X, y)


def run_single_backtest():
    """
    Runs a single backtest with a specific set of pre-defined parameters.
    This is useful for analyzing a single run without Optuna.
    """
    print("\n--- Running Single Backtest with Specific Parameters ---")

    # 1. Load Data
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
    )

    # 2. Define Parameters
    params = {
        "peak_method": "pct_change_on_ao",
        "peak_distance": 4,
        "peak_threshold": 0.5832,
        "corr_threshold": 0.01,
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 9,
        "subsample": 0.57,
        "colsample_bytree": 0.92,
        "gamma": 1.19e-5,
        "min_child_weight": 10,
        "refit_every": 24 * 7,
    }
    print("Using parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 3. Create Target Variable
    reversal_data = create_ao_target(
        data.copy(),
        method=params["peak_method"],
        peak_distance=params["peak_distance"],
        peak_threshold=params["peak_threshold"],
    )
    y = reversal_data["target"]
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})

    # 4. Create Features
    features_df = create_features(data)
    X = features_df.loc[reversal_data.index]
    X = X.loc[:, (X != X.iloc[0]).any()]

    # 5. Feature Selection
    selected_cols = select_features(X, y, corr_threshold=params["corr_threshold"])
    if selected_cols:
        X = X[selected_cols]
    print(selected_cols)

    if X.empty:
        print("Feature set is empty. Cannot run backtest.")
        return

    # 6. Setup and Run Backtest
    model_params = {
        "objective": "multi:softmax",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "gamma": params["gamma"],
        "min_child_weight": params["min_child_weight"],
        "random_state": 42,
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**model_params)

    manual_backtest(
        X, y_mapped, model, test_size=0.3, refit_every=params["refit_every"]
    )


def train_and_evaluate():
    """
    Trains an XGBoost model on 70% of the data and evaluates it on the remaining 30%.
    Prints a confusion matrix and classification report.
    """
    print("\n--- Training and Evaluating XGBoost on a 70/30 Split ---")

    price_change_bars = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT_price_change_bars_0_32.csv"
    # 1. Load Data
    data = fetch_historical_data(
        data_path=price_change_bars, start_date="2022-01-01T00:00:00Z"
    )

    # 2. Define Parameters (using a good set from run_single_backtest)
    params = {
        "peak_method": "pct_change_on_ao",
        "peak_distance": 4,
        "peak_threshold": 0.5832,
        "corr_threshold": 0.01,
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 9,
        "subsample": 0.57,
        "colsample_bytree": 0.92,
        "gamma": 1.19e-5,
        "min_child_weight": 10,
    }
    print("Using parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 3. Create Target and Features
    reversal_data = create_ao_target(
        data.copy(),
        method=params["peak_method"],
        peak_distance=params["peak_distance"],
        peak_threshold=params["peak_threshold"],
    )
    y = reversal_data["target"]
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})

    features_df = create_features(data)
    X = features_df.loc[reversal_data.index]
    X = X.loc[:, (X != X.iloc[0]).any()]

    # 4. Feature Selection
    selected_cols = select_features(X, y, corr_threshold=params["corr_threshold"])
    if selected_cols:
        X = X[selected_cols]

    if X.empty:
        print("Feature set is empty. Cannot run evaluation.")
        return

    # 5. Split data into training and testing sets
    split_index = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y_mapped.iloc[:split_index], y_mapped.iloc[split_index:]
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 6. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Calculate sample weights for class imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = y_train.map(class_weight_dict).to_numpy()

    # 8. Train XGBoost model on GPU
    model_params = {
        "objective": "multi:softmax",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "gamma": params["gamma"],
        "min_child_weight": params["min_child_weight"],
        "random_state": 42,
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**model_params)

    print("Training model...")
    X_train_gpu = cp.asarray(X_train_scaled)
    y_train_gpu = cp.asarray(y_train)
    sample_weights_gpu = cp.asarray(sample_weights)
    model.fit(X_train_gpu, y_train_gpu, sample_weight=sample_weights_gpu)

    # 9. Make predictions on the test set
    print("Making predictions on the test set...")
    X_test_gpu = cp.asarray(X_test_scaled)
    y_pred_gpu = model.predict(X_test_gpu)
    y_pred = cp.asnumpy(y_pred_gpu)

    # 10. Evaluate and print results
    print("\n--- Model Evaluation ---")
    target_names = ["Bottom (-1)", "Neutral (0)", "Top (1)"]
    labels = [0, 1, 2]  # Mapped labels

    # Classification Report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, labels=labels, target_names=target_names, zero_division=0
        )
    )

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.show()


if __name__ == "__main__":
    # main()  # Uncomment to run the Optuna study
    # run_single_backtest()
    train_and_evaluate()
