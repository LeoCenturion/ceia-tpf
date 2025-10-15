import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from autogluon.multimodal import MultiModalPredictor

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
    k_percent = 100 * ((close - low_n) / (high_n - low_n).replace(0, 1e-9))
    d_percent = sma(k_percent, d_n)
    return pd.DataFrame({'%K': k_percent, '%D': d_percent})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h_minus_l = high - low
    h_minus_pc = abs(high - close.shift(1))
    l_minus_pc = abs(low - close.shift(1))
    tr = pd.concat([h_minus_l, h_minus_pc, l_minus_pc], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return ewm(tr, span=2 * n - 1)

def willr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    high_n = high.rolling(n).max()
    low_n = low.rolling(n).min()
    wr = -100 * (high_n - close) / (high_n - low_n).replace(0, 1e-9)
    return wr

def roc(close: pd.Series, n: int = 10) -> pd.Series:
    return (close.diff(n) / close.shift(n)).replace([np.inf, -np.inf], 0) * 100

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a set of technical analysis features.
    """
    features = pd.DataFrame(index=df.index)
    close_pct = df['Close'].pct_change().fillna(0)
    features['pct_change'] = close_pct
    features['RSI'] = rsi_indicator(df['Close'], n=14)
    features['AO'] = awesome_oscillator(df['High'], df['Low'])
    features['WR'] = willr(df['High'], df['Low'], df['Close'])
    features['ROC'] = roc(df['Close'])
    stoch_price = stochastic_oscillator(df['High'], df['Low'], df['Close'])
    features['Stoch_K'] = stoch_price['%K']
    features['Stoch_D'] = stoch_price['%D']
    macd_price_df = macd(df['Close'])
    features['MACD'] = macd_price_df['MACD']
    features['MACD_Signal'] = macd_price_df['Signal']
    features['MACD_Hist'] = macd_price_df['Hist']
    if 'Volume' in df.columns:
        features['MFI'] = mfi(df['High'], df['Low'], df['Close'], df['Volume'], n=14)
    bbands = bollinger_bands(df['Close'])
    if bbands is not None and not bbands.empty:
        features['BBP'] = bbands.get('BBP_20_2.0')
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.bfill(inplace=True)
    features.ffill(inplace=True)
    return features

def bollinger_bands(close: pd.Series, n: int = 20, std_dev: float = 2.0):
    sma_val = sma(close, n)
    std_val = std(close, n)
    upper = sma_val + std_dev * std_val
    lower = sma_val - std_dev * std_val
    bbp = (close - lower) / (upper - lower).replace(0, 1e-9)
    return pd.DataFrame({f'BBP_{n}_{std_dev}': bbp, f'BBU_{n}_{std_dev}': upper, f'BBL_{n}_{std_dev}': lower})


def create_target_variable(df: pd.DataFrame, method: str = 'ao_on_price', peak_distance: int = 1, peak_threshold: float = 0) -> pd.DataFrame:
    """
    Identifies local tops (1), bottoms (-1), and non-reversal points (0).
    """
    if method == 'ao_on_price':
        ao = awesome_oscillator(df['High'], df['Low'])
    else:
        raise ValueError(f"Invalid method '{method}' specified for create_target_variable")

    if ao is None or ao.isnull().all():
        df['target'] = 0
        return df

    peaks, _ = find_peaks(ao, distance=peak_distance, threshold=peak_threshold)
    troughs, _ = find_peaks(-ao, distance=peak_distance, threshold=peak_threshold)

    df['target'] = 0
    df.loc[df.index[peaks], 'target'] = 1
    df.loc[df.index[troughs], 'target'] = -1
    return df

def main():
    """
    Main function to run a classification task using AutoGluon's MultiModalPredictor
    with an FT-Transformer model.
    """
    # 1. Load Data
    print("Loading historical data...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        timeframe="1h",
        start_date="2022-01-01T00:00:00Z"
    )

    # 2. Create Target Variable
    print("Creating target variable (price reversals)...")
    reversal_data = create_target_variable(data.copy(), method='ao_on_price', peak_distance=4, peak_threshold=50)

    # 3. Create Features
    print("Creating features...")
    features_df = create_features(data)

    # 4. Combine features and target, and drop rows with missing values
    final_df = pd.concat([features_df, reversal_data['target']], axis=1).dropna()
    
    # Remap labels to be non-negative for some metrics and models
    final_df['target'] = final_df['target'].map({-1: 0, 0: 1, 1: 2})
    final_df['target'] = final_df['target'].astype('category')

    print(f"Dataset shape: {final_df.shape}")
    print("Target distribution:")
    print(final_df['target'].value_counts())

    # 5. Split data into training and testing sets (70/30 split)
    train_end_index = int(len(final_df) * 0.7)
    train_data = final_df.iloc[:train_end_index]
    test_data = final_df.iloc[train_end_index:]

    print(f"Training data points: {len(train_data)}")
    print(f"Testing data points: {len(test_data)}")

    # 6. Initialize and fit MultiModalPredictor
    print("\nInitializing and fitting MultiModalPredictor with FT-Transformer...")
    predictor = MultiModalPredictor(
        label='target',
        problem_type='multiclass',
        eval_metric='f1_macro'
    )
    predictor.fit(
        train_data,
        hyperparameters={
            'model.names': ['ft_transformer'],
            'env.per_gpu_batch_size': 128
        },
        time_limit=600  # 10-minute time limit
    )

    # 7. Evaluate the model on the test set
    print("\nEvaluating model on the test set...")
    scores = predictor.evaluate(test_data)
    print("Evaluation scores:")
    print(scores)

if __name__ == "__main__":
    main()
