import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import find_peaks
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

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies local tops (1), bottoms (-1), and non-reversal points (0)
    using the Awesome Oscillator (AO) on price percentage changes,
    and returns the DataFrame with a 'target' column.
    """
    # Calculate Awesome Oscillator on percentage change
    high_pct = df['High'].pct_change().fillna(0)
    low_pct = df['Low'].pct_change().fillna(0)
    ao = awesome_oscillator(high_pct, low_pct)

    if ao is None or ao.isnull().all():
        # If AO can't be calculated, label all points as neutral
        df['target'] = 0
        return df

    # Find peaks (tops) and troughs (bottoms) in the AO
    peaks, _ = find_peaks(ao)
    troughs, _ = find_peaks(-ao)

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


def manual_backtest(X: pd.DataFrame, y: pd.Series, model, test_size: float = 0.3):
    """
    Performs a manual walk-forward backtest with an expanding window.
    At each step in the test set, the model is retrained using all historical data
    up to that point before making a prediction for the next step.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        model: A scikit-learn compatible classifier.
        test_size (float): The proportion of the dataset to be used for testing.
    """
    split_index = int(len(X) * (1 - test_size))
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    y_pred = []

    print("Starting walk-forward backtest with retraining at each step...")
    # Walk forward
    for i in range(len(X_test)):
        current_split_index = split_index + i

        # Define current training data (expanding window)
        X_train_current = X.iloc[:current_split_index]
        y_train_current = y.iloc[:current_split_index]

        # Define current test sample
        X_test_current = X.iloc[current_split_index:current_split_index + 1]

        # Scale data
        scaler = StandardScaler()
        X_train_current_scaled = scaler.fit_transform(X_train_current)
        X_test_current_scaled = scaler.transform(X_test_current)

        # Retrain model
        model.fit(X_train_current_scaled, y_train_current)

        # Predict
        prediction = model.predict(X_test_current_scaled)
        y_pred.append(prediction[0])

        if (i + 1) % 50 == 0 or (i + 1) == len(X_test):
            print(f"Processed {i+1}/{len(X_test)} steps...")

    # Evaluate
    print("\n--- Backtest Classification Report ---")
    print(classification_report(y_test, y_pred, labels=[-1, 0, 1], target_names=['Bottom (-1)', 'Neutral (0)', 'Top (1)'], zero_division=0))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Backtest Accuracy: {accuracy:.2f}")

    return y_pred, y_test


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


def main():
    """
    Main function to run the Random Forest price reversal classification task.
    """
    # 1. Load Data
    print("Loading historical data...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z"
    ).iloc[-2000:]

    # 2. Create Target Variable
    print("Identifying tops and bottoms to create target variable...")
    reversal_data = create_target_variable(data.copy())

    # Optional: Plot the candlestick chart with identified reversal points for visualization
    plot_reversals_on_candlestick(data, reversal_data, sample_size=8000)

    if (reversal_data['target'] == 0).all():
        print("No reversal points (tops/bottoms) were identified. Exiting.")
        return

    y = reversal_data['target']

    # 3. Create Features for the entire dataset
    print("Generating technical analysis features...")
    features_df = create_features(data)
    
    # Align features with the reversal points
    X = features_df.loc[reversal_data.index]
    
    # Drop any columns that might be constant
    X = X.loc[:, (X != X.iloc[0]).any()]

    # 4. Feature Selection
    print("Performing feature selection...")
    selected_cols = select_features(X, y)

    if not selected_cols:
        print("No features met the high correlation criteria. Using all generated features instead.")
    else:
        X = X[selected_cols]
        
    print(f"Final features being used: {list(X.columns)}")

    print("\nTarget variable distribution:")
    print(y.value_counts())

    # 5. Run Backtest
    print("Running walk-forward backtest...")
    class_weights = {-1: 10, 1: 10, 0: 1}
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weights)
    manual_backtest(X, y, model, test_size=0.3)

if __name__ == "__main__":
    main()
