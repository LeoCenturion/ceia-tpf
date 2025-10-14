import pandas as pd
import numpy as np
import optuna
from functools import partial
from scipy.signal import find_peaks
from sklearn.metrics import classification_report
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

from backtest_utils import fetch_historical_data, sma
# Note: The model "amazon/chronos-bolt-base" was not found.
# Using "amazon/chronos-t5-base" via AutoGluon instead.
def awesome_oscillator(high: pd.Series, low: pd.Series, fast_period: int = 5, slow_period: int = 34) -> pd.Series:
    """Calculates the Awesome Oscillator."""
    median_price = (high + low) / 2
    ao = sma(median_price, fast_period) - sma(median_price, slow_period)
    return ao


def create_target_variable(df: pd.DataFrame, method: str = 'ao_on_pct_change', peak_distance: int = 1, peak_threshold: float = 0, std_fraction: float = 1.0) -> pd.DataFrame:
    """
    Identifies local tops (1), bottoms (-1), and non-reversal points (0)
    using different methods, and returns the DataFrame with a 'target' column.

    Methods:
    - 'ao_on_price': Awesome Oscillator on actual prices.
    - 'ao_on_pct_change': Awesome Oscillator on price percentage changes.
    - 'pct_change_on_ao': Percentage change of Awesome Oscillator on actual prices.
    - 'pct_change_std': Target based on closing price pct_change exceeding 1 std dev.
    """
    if method == 'pct_change_std':
        window = 24 * 7
        close_pct_change = df['Close'].pct_change()
        # The target is based on the NEXT period's price change.
        future_pct_change = close_pct_change.shift(-1)
        rolling_std = close_pct_change.rolling(window=window).std()

        df['target'] = 0
        df.loc[future_pct_change >= (rolling_std * std_fraction), 'target'] = 1
        df.loc[future_pct_change <= -(rolling_std * std_fraction), 'target'] = -1
        return df

    if method == 'ao_on_pct_change':
        # computing the peaks from the awesome oscillator from the pct_change of the values
        high_pct = df['High'].pct_change().fillna(0)
        low_pct = df['Low'].pct_change().fillna(0)
        ao = awesome_oscillator(high_pct, low_pct)
    elif method == 'ao_on_price':
        # computing the peaks from the awesome oscillator from the actual price values
        ao = awesome_oscillator(df['High'], df['Low'])
    elif method == 'pct_change_on_ao':
        # computing the peaks from the pct_change of the awesome oscillator from the actual price values
        ao_price = awesome_oscillator(df['High'], df['Low'])
        ao = ao_price.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    else:
        raise ValueError(f"Invalid method '{method}' specified for create_target_variable")


    if ao is None or ao.isnull().all():
        # If AO can't be calculated, label all points as neutral
        df['target'] = 0
        return df

    # Find peaks (tops) and troughs (bottoms) in the AO
    peaks, _ = find_peaks(ao, distance=peak_distance, threshold=peak_threshold)
    troughs, _ = find_peaks(-ao, distance=peak_distance, threshold=peak_threshold)

    # Create the target column, default to 0 (neutral)
    df['target'] = 0
    df.loc[df.index[peaks], 'target'] = 1  # Tops
    df.loc[df.index[troughs], 'target'] = -1 # Bottoms

    return df


# --- Part 2: Model Backtesting ---

def manual_backtest(df: pd.DataFrame, model_params: dict, trade_threshold: float, test_size: float = 0.3, refit_every: int = 24 * 7):
    """
    Performs a walk-forward backtest using AutoGluon TimeSeriesPredictor.
    """
    split_index = int(len(df) * (1 - test_size))
    test_data = df.iloc[split_index:]
    y_test = df['target'].iloc[split_index:]
    y_pred = []

    if len(test_data) < 2:
        return classification_report(y_test, y_pred, labels=[-1, 0, 1], target_names=['Bottom', 'Neutral', 'Top'], zero_division=0, output_dict=True)

    was_long = False
    predictor = None
    
    print("Starting walk-forward backtest...")
    for i in range(len(test_data)):
        current_split_index = split_index + i
        
        # Periodically refit the model on an expanding window
        if i % refit_every == 0:
            print(f"Refitting model at step {i+1}/{len(test_data)}...")
            train_data = df.iloc[:current_split_index]
            
            # Prepare data for AutoGluon
            train_data_ag = train_data[['Close']].copy()
            train_data_ag.reset_index(inplace=True)
            train_data_ag.rename(columns={'index': 'timestamp', 'Close': 'target'}, inplace=True)
            train_data_ag['item_id'] = 'BTC'
            train_ts_df = TimeSeriesDataFrame.from_data_frame(train_data_ag, id_column="item_id", timestamp_column="timestamp")
            
            predictor = TimeSeriesPredictor(
                prediction_length=1,
                eval_metric="MASE",
                target="target"
            )
            predictor.fit(
                train_ts_df,
                hyperparameters={"Chronos": model_params},
                time_limit=300 # 5 minute timeout for fitting
            )

        # Predict the next step
        if predictor is not None:
            # Predict requires the training data to know where to start forecasting from
            current_train_data = df.iloc[:current_split_index]
            current_train_ag = current_train_data[['Close']].copy()
            current_train_ag.reset_index(inplace=True)
            current_train_ag.rename(columns={'index': 'timestamp', 'Close': 'target'}, inplace=True)
            current_train_ag['item_id'] = 'BTC'
            current_ts_df = TimeSeriesDataFrame.from_data_frame(current_train_ag, id_column="item_id", timestamp_column="timestamp")

            forecast = predictor.predict(current_ts_df)
            predicted_price = forecast['mean'].iloc[0]
            current_price = current_train_data['Close'].iloc[-1]
            
            is_long = predicted_price > current_price * (1 + trade_threshold)

            # Generate reversal signal
            if is_long and not was_long:
                predicted_signal = -1  # Buy signal -> predicts a bottom
            elif not is_long and was_long:
                predicted_signal = 1   # Sell signal -> predicts a top
            else:
                predicted_signal = 0   # No change in position
            
            was_long = is_long
        else:
            predicted_signal = 0 # Cannot predict if model not trained

        y_pred.append(predicted_signal)
        
        if (i + 1) % 100 == 0 or (i + 1) == len(test_data):
            print(f"Processed {i+1}/{len(test_data)} steps...")

    print("\n--- Backtest Classification Report ---")
    report = classification_report(y_test, y_pred, labels=[-1, 0, 1], target_names=['Bottom', 'Neutral', 'Top'], zero_division=0, output_dict=True)
    print(classification_report(y_test, y_pred, labels=[-1, 0, 1], target_names=['Bottom', 'Neutral', 'Top'], zero_division=0))
    return report


# --- Part 3: Optuna Optimization ---

def objective(trial: optuna.Trial, hourly_data: pd.DataFrame, model_name: str) -> float:
    """
    Optuna objective function to tune hyperparameters for the Chronos strategy.
    """
    # --- 1. Define Hyperparameter Search Space ---
    # Peak detection hyperparameters
    peak_method = trial.suggest_categorical('peak_method', ['ao_on_price', 'pct_change_on_ao'])
    peak_distance = trial.suggest_int('peak_distance', 1, 5)
    if peak_method == 'ao_on_price':
        peak_threshold = trial.suggest_float('peak_threshold', 0.0, 100.0)
    else: # pct_change_on_ao
        peak_threshold = trial.suggest_float('peak_threshold', 0.0, 5)

    # Chronos model hyperparameters
    context_length = trial.suggest_int('context_length', 60, 300, step=30)
    
    # Backtesting hyperparameters
    trade_threshold = trial.suggest_float('trade_threshold', 0.0005, 0.01, log=True)
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: {trial.params}")
    
    # --- 2. Run the ML Pipeline ---
    # Create Target Variable
    reversal_data = create_target_variable(
        hourly_data.copy(),
        method=peak_method,
        peak_distance=peak_distance,
        peak_threshold=peak_threshold
    )

    # Prune trial if not enough reversal points are found
    if reversal_data['target'].nunique() < 3 or reversal_data['target'].value_counts().get(1, 0) < 5 or reversal_data['target'].value_counts().get(-1, 0) < 5:
        print("Not enough reversal points found with these parameters. Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    # Setup Chronos parameters for AutoGluon
    model_params = {
        'model_path': model_name,
        'context_length': context_length,
    }

    # Run Backtest
    report = manual_backtest(
        reversal_data,
        model_params=model_params,
        trade_threshold=trade_threshold
    )
    
    # === 3. Calculate and Return the Objective Metric ===
    f1_top = report.get('Top', {}).get('f1-score', 0.0)
    f1_bottom = report.get('Bottom', {}).get('f1-score', 0.0)
    
    # We want to maximize the average F1 score for identifying tops and bottoms
    objective_value = (f1_top + f1_bottom) / 2
    
    trial.set_user_attr("f1_top", f1_top)
    trial.set_user_attr("f1_bottom", f1_bottom)

    print(f"Trial {trial.number} finished. Avg F1 (Top/Bottom): {objective_value:.4f}")
    return objective_value


def main():
    """
    Main function to run the Optuna hyperparameter optimization study.
    """
    N_TRIALS = 30
    MODEL_NAME = "amazon/chronos-t5-base" # Use 't5' for better performance
    
    # 1. Load Data
    print("Loading 1-hour historical data...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z"
    )
    
    # 2. Setup and Run Optuna Study
    db_file_name = "optuna-study"
    study_name_in_db = 'chronos_autogluon_v1'
    storage_name = f"sqlite:///{db_file_name}.db"
    print(f"Starting Optuna study: '{study_name_in_db}'. Storage: {storage_name}")

    objective_with_data = partial(objective, hourly_data=data, model_name=MODEL_NAME)
    
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name_in_db,
        storage=storage_name,
        load_if_exists=True
    )
    # n_jobs=1 is recommended for GPU-based models to avoid memory conflicts
    study.optimize(objective_with_data, n_trials=N_TRIALS, n_jobs=1)
    
    # 3. Print Study Results
    print("\n--- Optuna Study Best Results ---")
    try:
        best_trial = study.best_trial
        print(f"Best trial value (Average F1 Score): {best_trial.value}")
        print("Best parameters found:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print("Associated metrics:")
        print(f"  F1 Score (Top): {best_trial.user_attrs.get('f1_top', 'N/A'):.4f}")
        print(f"  F1 Score (Bottom): {best_trial.user_attrs.get('f1_bottom', 'N/A'):.4f}")
    except ValueError:
        print("No successful trials were completed.")


if __name__ == "__main__":
    main()
