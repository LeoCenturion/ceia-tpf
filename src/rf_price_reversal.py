import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

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
    Creates a set of technical analysis features.
    """
    features = pd.DataFrame(index=df.index)

    # Price and percentage change
    features['Close'] = df['Close']
    features['pct_change'] = df['Close'].pct_change()

    # Momentum Indicators
    features['RSI'] = rsi_indicator(df['Close'], n=14)
    stoch = stochastic_oscillator(df['High'], df['Low'], df['Close'])
    features['Stoch_K'] = stoch['%K']
    features['Stoch_D'] = stoch['%D']

    # Trend Indicators
    macd_df = macd(df['Close'])
    features['MACD'] = macd_df['MACD']
    features['MACD_Signal'] = macd_df['Signal']
    features['MACD_Hist'] = macd_df['Hist']

    # Volume Indicators
    features['MFI'] = mfi(df['High'], df['Low'], df['Close'], df['Volume'], n=14)

    # Volatility Indicators
    sma20 = sma(df['Close'], 20)
    std20 = std(df['Close'], 20)
    features['BB_Upper'] = sma20 + (std20 * 2)
    features['BB_Lower'] = sma20 - (std20 * 2)
    features['BB_Width'] = (features['BB_Upper'] - features['BB_Lower']) / sma20

    # Fill NaN values that might have been generated
    features.bfill(inplace=True)
    features.ffill(inplace=True)

    return features

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies local tops (1) and bottoms (0) using the Awesome Oscillator (AO)
    and returns a DataFrame containing only the rows with these reversal points.
    """
    # Calculate Awesome Oscillator
    ao = awesome_oscillator(df['High'], df['Low'])
    if ao is None or ao.isnull().all():
        # Return an empty DataFrame if AO can't be calculated
        return pd.DataFrame(columns=df.columns.tolist() + ['target'])

    # Find peaks (tops) and troughs (bottoms) in the AO
    peaks, _ = find_peaks(ao)
    troughs, _ = find_peaks(-ao)

    # Create the target column, initially with NaNs
    df['target'] = np.nan
    df.loc[df.index[peaks], 'target'] = 1  # Tops
    df.loc[df.index[troughs], 'target'] = 0 # Bottoms

    # We are only interested in classifying the reversal points
    reversal_points_df = df.dropna(subset=['target'])

    return reversal_points_df

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

# AI write a backtesting function that manually backtest given a dataframe ...
# for i in range(start, len(self._data)):
#                data._set_length(i + 1)
# in each iteration it should retrain the model (if needed) and predict the target.
# run a classification report in the end AI!

def main():
    """
    Main function to run the Random Forest price reversal classification task.
    """
    # 1. Load Data
    print("Loading historical data...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z"
    )

    # 2. Create Target Variable
    print("Identifying tops and bottoms to create target variable...")
    reversal_data = create_target_variable(data.copy())
    
    if reversal_data.empty:
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

    # 5. Split Data (Sequential Split)
    # Using a sequential split to avoid training on future data.
    test_size = 0.3
    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 6. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Train Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # 8. Evaluate Model
    print("Evaluating model performance...")
    y_pred = model.predict(X_test_scaled)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bottom (0)', 'Top (1)'], zero_division=0))
    
    # Print overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
