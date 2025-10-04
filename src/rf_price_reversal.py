import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import optuna
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import mplfinance as mpf

from backtest_utils import fetch_historical_data, sma, ewm, std, rsi_indicator


def awesome_oscillator(high: pd.Series, low: pd.Series, fast_period: int = 5, slow_period: int = 34) -> pd.Series:
    """Calculates the Awesome Oscillator."""
    median_price = (high + low) / 2
    ao = sma(median_price, fast_period) - sma(median_price, slow_period)
    return ao


def macd(close: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """Calculates MACD, Signal Line, and Histogram."""
    ema_fast = ewm(close, span=fast_period)
    ema_slow = ewm(close, span=slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = ewm(macd_line, span=signal_period)
    histogram = macd_line - signal_line
    return pd.DataFrame({'MACD': macd_line, 'Signal': signal_line, 'Hist': histogram})


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    """Calculates the Money Flow Index."""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0), index=typical_price.index)
    negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0), index=typical_price.index)

    positive_mf = positive_flow.rolling(window=n, min_periods=0).sum()
    negative_mf = negative_flow.rolling(window=n, min_periods=0).sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
    mfi.replace([np.inf, -np.inf], 100, inplace=True)
    return mfi


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, d_n: int = 3) -> pd.DataFrame:
    """Calculates the Stochastic Oscillator (%K and %D)."""
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    k_percent = 100 * ((close - low_n) / (high_n - low_n))
    d_percent = sma(k_percent, d_n)
    return pd.DataFrame({'%K': k_percent, '%D': d_percent})


# AI add the following features if they're not already
# average volume
# fluctuation kcp
# Volume
# fluctuation bbp
# trend macd
# fluctuation DCP
# trend ADX
# trend difference MACD
# trend difference VORTEX
# VORTEX trend
# trend AROON
# trend CCI
# momentum RSI
# trend STC
# momentum UO
# momentum TSI
# momentum STOCH SIGNAL
# stochastic momentum
# momentum AO
# momentum WR
# momentum ROC
# above EMA 10,20,40,60
# Relative stregth index
# above EMA 15,30,50
# AI!

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a set of technical analysis features based on percentage changes.
    """
    features = pd.DataFrame(index=df.index)

    # Price and percentage change
    high_pct = df['High'].pct_change().fillna(0)
    low_pct = df['Low'].pct_change().fillna(0)
    close_pct = df['Close'].pct_change().fillna(0)
    features['pct_change'] = close_pct

    # Momentum Indicators
    features['RSI'] = rsi_indicator(close_pct, n=14)
    stoch = stochastic_oscillator(high_pct, low_pct, close_pct)
    features['Stoch_K'] = stoch['%K']
    features['Stoch_D'] = stoch['%D']

    # Trend Indicators
    macd_df = macd(close_pct)
    features['MACD'] = macd_df['MACD']
    features['MACD_Signal'] = macd_df['Signal']
    features['MACD_Hist'] = macd_df['Hist']

    # Volume Indicators
    features['MFI'] = mfi(high_pct, low_pct, close_pct, df['Volume'], n=14)

    # Volatility Indicators
    sma20 = sma(close_pct, 20)
    std20 = std(close_pct, 20)
    features['BB_Upper'] = sma20 + (std20 * 2)
    features['BB_Lower'] = sma20 - (std20 * 2)
    features['BB_Width'] = (features['BB_Upper'] - features['BB_Lower']) / sma20

    # Fill NaN values that might have been generated
    features.bfill(inplace=True)
    features.ffill(inplace=True)

    return features

def create_target_variable(df: pd.DataFrame, method: str = 'ao_on_pct_change', peak_distance: int = 1, peak_threshold: float = 0) -> pd.DataFrame:
    """
    Identifies local tops (1), bottoms (-1), and non-reversal points (0)
    using different methods, and returns the DataFrame with a 'target' column.

    Methods:
    - 'ao_on_price': Awesome Oscillator on actual prices.
    - 'ao_on_pct_change': Awesome Oscillator on price percentage changes.
    - 'pct_change_on_ao': Percentage change of Awesome Oscillator on actual prices.
    """
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

def select_features(X: pd.DataFrame, y: pd.Series, corr_threshold=0.7, p_value_threshold=0.05) -> list:
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

    print(f"Selected {len(selected_features)} features out of {len(X.columns)} based on correlation criteria.")
    return selected_features

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

            # Scale training data and retrain model
            X_train_current_scaled = scaler.fit_transform(X_train_current)
            model.fit(X_train_current_scaled, y_train_current)

        # Get current test sample and scale it using the latest scaler
        X_test_current = X.iloc[current_split_index:current_split_index + 1]
        X_test_current_scaled = scaler.transform(X_test_current)

        # Predict
        prediction = model.predict(X_test_current_scaled)
        y_pred.append(prediction[0])

        if (i + 1) % 50 == 0 or (i + 1) == len(X_test):
            print(f"Processed {i+1}/{len(X_test)} steps...")

    # Evaluate
    print("\n--- Backtest Classification Report ---")
    report_str = classification_report(y_test, y_pred, labels=[-1, 0, 1], target_names=['Bottom (-1)', 'Neutral (0)', 'Top (1)'], zero_division=0)
    print(report_str)
    report_dict = classification_report(y_test, y_pred, labels=[-1, 0, 1], target_names=['Bottom (-1)', 'Neutral (0)', 'Top (1)'], zero_division=0, output_dict=True)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Backtest Accuracy: {accuracy:.2f}")

    return y_pred, y_test, report_dict


def plot_reversals_on_candlestick(data: pd.DataFrame, reversal_points: pd.DataFrame, sample_size: int = None):
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
        plot_reversal_points = reversal_points[reversal_points.index >= plot_data.index[0]]

    # Create series for plotting markers
    tops = plot_reversal_points[plot_reversal_points['target'] == 1]
    bottoms = plot_reversal_points[plot_reversal_points['target'] == -1]

    # Place markers slightly above highs for tops and below lows for bottoms
    top_markers = pd.Series(np.nan, index=plot_data.index)
    bottom_markers = pd.Series(np.nan, index=plot_data.index)

    top_indices = tops.index.intersection(plot_data.index)
    bottom_indices = bottoms.index.intersection(plot_data.index)

    # Check if there are any tops/bottoms to plot to avoid errors on empty access
    if not top_indices.empty:
        top_markers.loc[top_indices] = plot_data.loc[top_indices, 'High'] * 1.01
    if not bottom_indices.empty:
        bottom_markers.loc[bottom_indices] = plot_data.loc[bottom_indices, 'Low'] * 0.99

    # Create addplots for mplfinance
    addplots = [
        mpf.make_addplot(top_markers, type='scatter', marker='^', color='green', markersize=100),
        mpf.make_addplot(bottom_markers, type='scatter', marker='v', color='red', markersize=100)
    ]

    title = 'Candlestick Chart with Tops and Bottoms'
    if sample_size:
        title += f' (Last {sample_size} hours)'

    mpf.plot(plot_data, type='candle', style='yahoo',
             title=title,
             ylabel='Price ($)',
             addplot=addplots,
             figsize=(15, 7),
             volume=True,
             panel_ratios=(3, 1))

# AI resume running trials
def objective(trial: optuna.Trial, data: pd.DataFrame) -> float:
    """
    Optuna objective function to tune hyperparameters for the RF price reversal model.
    """
    # === 1. Define Hyperparameter Search Space ===
    # Peak detection hyperparameters
    peak_method = trial.suggest_categorical('peak_method', ['ao_on_price', 'ao_on_pct_change', 'pct_change_on_ao'])
    peak_distance = trial.suggest_int('peak_distance', 1, 5)

    if peak_method == 'ao_on_price':
        # Threshold is a difference, so values can be smaller than absolute price
        peak_threshold = trial.suggest_float('peak_threshold', 0.0, 100.0)
    elif peak_method == 'pct_change_on_ao':
        peak_threshold = trial.suggest_float('peak_threshold', 0.0, 5)
    else:  # For pct_change based methods, the scale is much smaller
        peak_threshold = trial.suggest_float('peak_threshold', 0.0, 0.005)

    # Feature selection hyperparameter
    corr_threshold = trial.suggest_float('corr_threshold', 0.1, 0.7)

    # Model hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    refit_every = 24 * 7

    # === 2. Run the ML Pipeline ===
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: {trial.params}")

    # Create Target Variable
    reversal_data = create_target_variable(data.copy(), method=peak_method, peak_distance=peak_distance, peak_threshold=peak_threshold)

    # Prune trial if not enough reversal points are found
    if reversal_data['target'].nunique() < 3 or reversal_data['target'].value_counts().get(1, 0) < 5 or reversal_data['target'].value_counts().get(-1, 0) < 5:
        print("Not enough reversal points found with these parameters. Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    y = reversal_data['target']

    # Create Features
    features_df = create_features(data)
    X = features_df.loc[reversal_data.index]
    X = X.loc[:, (X != X.iloc[0]).any()]  # Drop constant columns

    # Feature Selection
    selected_cols = select_features(X, y, corr_threshold=corr_threshold)
    if selected_cols:
        X = X[selected_cols]
    
    # Run Backtest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    _, _, report = manual_backtest(X, y, model, test_size=0.3, refit_every=refit_every)

    # === 3. Calculate and Return the Objective Metric ===
    f1_top = report.get('Top (1)', {}).get('f1-score', 0.0)
    f1_bottom = report.get('Bottom (-1)', {}).get('f1-score', 0.0)
    
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
        start_date="2022-01-01T00:00:00Z"
    )

    # 2. Setup and Run Optuna Study
    db_file_name = "optuna-study"
    study_name_in_db = 'rf price reversal'
    storage_name = f"sqlite:///{db_file_name}.db"
    
    print(f"Starting Optuna study: '{study_name_in_db}'. Storage: {storage_name}")
    
    # Use a partial function to pass the loaded data to the objective function
    objective_with_data = partial(objective, data=data)

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name_in_db,
        storage=storage_name,
        load_if_exists=True
    )

    study.optimize(objective_with_data, n_trials=N_TRIALS, n_jobs=-1)
    # 3. Print Study Results
    print("\n--- Optuna Study Best Results ---")
    print(f"Best trial value (Average F1 Score): {study.best_value}")
    print("Best parameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
