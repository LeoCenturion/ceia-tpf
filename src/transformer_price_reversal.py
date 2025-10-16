import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from autogluon.multimodal import MultiModalPredictor
import optuna
from functools import partial
from sklearn.metrics import classification_report

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
    elif method == 'ao_on_pct_change':
        high_pct = df['High'].pct_change().fillna(0)
        low_pct = df['Low'].pct_change().fillna(0)
        ao = awesome_oscillator(high_pct, low_pct)
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


def manual_backtest_autogluon(
    df: pd.DataFrame,
    hyperparameters: dict,
    test_size: float = 0.3,
    refit_every: int = 24 * 7
):
    """
    Performs a walk-forward backtest for an AutoGluon MultiModalPredictor.
    The model is refit every `refit_every` steps on an expanding window.
    """
    split_index = int(len(df) * (1 - test_size))
    test_data = df.iloc[split_index:]
    y_test = test_data['target']
    y_pred = []
    predictor = None

    print("Starting walk-forward backtest...")
    for i in range(len(test_data)):
        current_split_index = split_index + i
        
        if i % refit_every == 0:
            print(f"Refitting model at step {i+1}/{len(test_data)}...")
            train_data_current = df.iloc[:current_split_index]

            predictor = MultiModalPredictor(
                label='target',
                problem_type='multiclass',
                eval_metric='f1_macro'
            )
            # Suppress verbose logs from AutoGluon during backtest fitting
            predictor.fit(
                train_data_current,
                hyperparameters=hyperparameters,
                time_limit=600
            )

        if predictor:
            X_test_current = test_data.drop(columns=['target']).iloc[i:i+1]
            prediction = predictor.predict(X_test_current)
            y_pred.append(prediction.iloc[0])
        else:
            # Should not happen, but as a fallback, predict neutral
            y_pred.append(1)

        if (i + 1) % 50 == 0 or (i + 1) == len(test_data):
            print(f"Processed {i+1}/{len(test_data)} steps...")
    
    labels = [0, 1, 2]
    target_names = ['Bottom (-1)', 'Neutral (0)', 'Top (1)']
    
    print("\n--- Backtest Classification Report ---")
    report_str = classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0)
    print(report_str)
    report_dict = classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0, output_dict=True)
    
    return report_dict


def objective(trial: optuna.Trial, data: pd.DataFrame) -> float:
    """
    Optuna objective function to tune hyperparameters for the FT-Transformer price reversal model.
    """
    # === 1. Define Hyperparameter Search Space ===
    # Peak detection hyperparameters
    peak_distance = trial.suggest_int('peak_distance', 1, 5)
    peak_threshold = trial.suggest_float('peak_threshold', 0.0, 0.005)

    # Model hyperparameters for FT-Transformer
    num_blocks = trial.suggest_int('num_blocks', 2, 6)
    hidden_size = trial.suggest_categorical('hidden_size', [128, 192, 256])

    # Optimization hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    # Backtesting hyperparameter
    refit_every = 24 * 7

    # === 2. Run the ML Pipeline ===
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: {trial.params}")

    # Create Target Variable
    reversal_data = create_target_variable(
        data.copy(),
        method='ao_on_pct_change',
        peak_distance=peak_distance,
        peak_threshold=peak_threshold
    )
    
    # Prune trial if not enough reversal points are found
    if reversal_data['target'].nunique() < 3 or reversal_data['target'].value_counts().get(1, 0) < 5 or reversal_data['target'].value_counts().get(-1, 0) < 5:
        print("Not enough reversal points found with these parameters. Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    # Create Features
    features_df = create_features(data)
    final_df = pd.concat([features_df, reversal_data['target']], axis=1).dropna()
    final_df['target'] = final_df['target'].map({-1: 0, 0: 1, 1: 2}).astype('category')
    
    # Prune trial if initial training set is not representative
    split_index = int(len(final_df) * (1 - 0.3)) # Corresponds to test_size in manual_backtest
    if final_df.iloc[:split_index]['target'].nunique() < 3:
        print("Initial training set does not contain all 3 classes. Pruning trial.")
        raise optuna.exceptions.TrialPruned()
    
    hyperparameters = {
        'model.names': ['ft_transformer'],
        'model.ft_transformer.num_blocks': num_blocks,
        'model.ft_transformer.hidden_size': hidden_size,
        'model.ft_transformer.token_dim': hidden_size,
        'model.ft_transformer.ffn_hidden_size': hidden_size * 2,
        'optim.lr': lr,
        'optim.weight_decay': weight_decay,
        'env.per_gpu_batch_size': 128
    }

    try:
        # Run Backtest
        report = manual_backtest_autogluon(
            final_df,
            hyperparameters=hyperparameters,
            test_size=0.3,
            refit_every=refit_every
        )

        # === 3. Calculate and Return the Objective Metric ===
        f1_top = report.get('Top (1)', {}).get('f1-score', 0.0)
        f1_bottom = report.get('Bottom (-1)', {}).get('f1-score', 0.0)
        
        # We want to maximize the average F1 score for identifying tops and bottoms
        objective_value = (f1_top + f1_bottom) / 2

    except Exception as e:
        print(f"Trial {trial.number} failed with an exception: {e}")
        objective_value = 0.0  # Penalize failed trials

    print(f"Trial {trial.number} finished. Avg F1 (Top/Bottom): {objective_value:.4f}")
    return objective_value


def run_single_backtest():
    """
    Runs a single backtest with a specific set of pre-defined parameters.
    This is useful for analyzing a single run without Optuna.
    """
    print("\n--- Running Single Backtest with Specific Parameters ---")

    # 1. Load Data
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        timeframe="1h",
        start_date="2022-01-01T00:00:00Z"
    )

    # 2. Define Parameters
    params = {
        'peak_distance': 4,
        'peak_threshold': 0.001,
        'num_blocks': 4,
        'hidden_size': 192,
        'lr': 0.0005,
        'weight_decay': 1e-4,
        'refit_every': 24 * 7
    }
    print("Using parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 3. Create Target Variable
    reversal_data = create_target_variable(
        data.copy(),
        method='ao_on_pct_change',
        peak_distance=params['peak_distance'],
        peak_threshold=params['peak_threshold']
    )

    # 4. Create Features
    features_df = create_features(data)
    final_df = pd.concat([features_df, reversal_data['target']], axis=1).dropna()
    final_df['target'] = final_df['target'].map({-1: 0, 0: 1, 1: 2}).astype('category')
    print("\nTarget class distribution in the full dataset:")
    print(final_df['target'].value_counts())

    # 5. Split data
    train_end_index = int(len(final_df) * 0.7)
    train_data = final_df.iloc[:train_end_index]
    test_data = final_df.iloc[train_end_index:]
    print(f"Training data points: {len(train_data)}")
    print(f"Testing data points: {len(test_data)}")


    hyperparameters = {
        'model.names': ['ft_transformer'],
        'model.ft_transformer.num_blocks': params['num_blocks'],
        'model.ft_transformer.hidden_size': params['hidden_size'],
        'model.ft_transformer.token_dim': params['hidden_size'],
        'model.ft_transformer.ffn_hidden_size': params['hidden_size'] * 2,
        'optim.lr': params['lr'],
        'optim.weight_decay': params['weight_decay'],
        'env.per_gpu_batch_size': 128
    }

    predictor = MultiModalPredictor(
        label='target',
        problem_type='multiclass',
        eval_metric='f1_macro'
    )
    predictor.fit(
        train_data,
        hyperparameters=hyperparameters,
        time_limit=6000
    )

    print("\n--- Evaluating on Test Set ---")
    y_pred = predictor.predict(test_data.drop(columns=['target']))
    y_true = test_data['target']

    labels = [0, 1, 2]
    target_names = ['Bottom (-1)', 'Neutral (0)', 'Top (1)']

    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0))

    report_dict = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0, output_dict=True)
    f1_top = report_dict.get('Top (1)', {}).get('f1-score', 0.0)
    f1_bottom = report_dict.get('Bottom (-1)', {}).get('f1-score', 0.0)
    avg_f1 = (f1_top + f1_bottom) / 2

    print(f"\nAverage F1 Score (Top/Bottom): {avg_f1:.4f}\n")

    scores = predictor.evaluate(test_data)
    print("AutoGluon Evaluation Scores:")
    print(scores)


def run_regression_evaluation():
    """
    Evaluates the FT-Transformer model on a regression task to predict
    the next period's percentage change in closing price.
    """
    print("\n--- Running Regression Evaluation to Predict Pct Change ---")

    # 1. Load Data
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        timeframe="1h",
        start_date="2022-01-01T00:00:00Z"
    )

    # 2. Create Features
    print("Creating features...")
    features_df = create_features(data)

    # 3. Create Target Variable (next period's pct_change)
    print("Creating regression target (next pct_change)...")
    target = data['Close'].pct_change().shift(-1)
    target.name = 'target_pct_change'

    # 4. Combine features and target, and drop rows with missing values
    final_df = pd.concat([features_df, target], axis=1).dropna()
    print(f"Dataset shape: {final_df.shape}")

    # 5. Split data
    train_end_index = int(len(final_df) * 0.7)
    train_data = final_df.iloc[:train_end_index]
    test_data = final_df.iloc[train_end_index:]
    print(f"Training data points: {len(train_data)}")
    print(f"Testing data points: {len(test_data)}")

    # 6. Setup and run regression evaluation
    # Using similar FT-Transformer params from the classification task
    hyperparameters = {
        'model.names': ['ft_transformer'],
        'model.ft_transformer.num_blocks': 4,
        'model.ft_transformer.hidden_size': 192,
        'model.ft_transformer.token_dim': 192,
        'model.ft_transformer.ffn_hidden_size': 192 * 2,
        'optim.lr': 0.0005,
        'optim.weight_decay': 1e-4,
        'env.per_gpu_batch_size': 128
    }

    predictor = MultiModalPredictor(
        label='target_pct_change',
        problem_type='regression',
        eval_metric='r2'
    )
    predictor.fit(
        train_data,
        hyperparameters=hyperparameters,
        time_limit=600
    )

    print("\n--- Evaluating Regression Model on Test Set ---")
    scores = predictor.evaluate(test_data)
    print("Evaluation scores (R^2, RMSE, etc.):")
    print(scores)


def main():
    """
    Main function to run the Optuna hyperparameter optimization study.
    """
    N_TRIALS = 1

    # 1. Load Data
    print("Loading historical data...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        timeframe="1h",
        start_date="2022-01-01T00:00:00Z"
    )

    # 2. Setup and Run Optuna Study
    db_file_name = "optuna-study"
    study_name_in_db = 'transformer_reversal_v0.2'
    storage_name = f"sqlite:///{db_file_name}.db"

    print(f"Starting Optuna study: '{study_name_in_db}'. Storage: {storage_name}")

    objective_with_data = partial(objective, data=data)

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name_in_db,
        storage=storage_name,
        load_if_exists=True
    )

    study.optimize(objective_with_data, n_trials=N_TRIALS, n_jobs=1) # n_jobs=1 for GPU usage
    
    # 3. Print Study Results
    print("\n--- Optuna Study Best Results ---")
    try:
        best_trial = study.best_trial
        print(f"Best trial value (F1 Macro): {best_trial.value}")
        print("Best parameters found:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    except ValueError:
        print("No successful trials were completed.")

if __name__ == "__main__":
    # main()  # Uncomment to run the Optuna study
    run_single_backtest()
    # run_regression_evaluation()
