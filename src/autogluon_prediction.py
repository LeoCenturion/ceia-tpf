def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a set of technical analysis features based on percentage changes and raw price data.
    """
    features = pd.DataFrame(index=df.index)

    # --- Original features based on percentage changes ---
    open_pct = df["Open"].pct_change().fillna(0)
    high_pct = df["High"].pct_change().fillna(0)
    low_pct = df["Low"].pct_change().fillna(0)
    close_pct = df["Close"].pct_change().fillna(0)
    features["pct_change"] = close_pct
    features["RSI_pct"] = rsi_indicator(close_pct, n=14)
    stoch_pct = stochastic_oscillator(high_pct, low_pct, close_pct)
    features["Stoch_K_pct"] = stoch_pct["%K"]
    features["Stoch_D_pct"] = stoch_pct["%D"]
    macd_pct_df = macd(close_pct)
    features["MACD_pct"] = macd_pct_df["MACD"]
    features["MACD_Signal_pct"] = macd_pct_df["Signal"]
    features["MACD_Hist_pct"] = macd_pct_df["Hist"]
    if "Volume" in df.columns:
        features["MFI_pct"] = mfi(high_pct, low_pct, close_pct, df["Volume"], n=14)
    sma20_pct = sma(close_pct, 20)
    std20_pct = std(close_pct, 20)
    features["BB_Upper_pct"] = sma20_pct + (std20_pct * 2)
    features["BB_Lower_pct"] = sma20_pct - (std20_pct * 2)
    features["BB_Width_pct"] = (
        features["BB_Upper_pct"] - features["BB_Lower_pct"]
    ) / sma20_pct

    # Lagged pct_change features
    for lag in range(1, 6):
        features[f"open_pct_lag_{lag}"] = open_pct.shift(lag)
        features[f"high_pct_lag_{lag}"] = high_pct.shift(lag)
        features[f"low_pct_lag_{lag}"] = low_pct.shift(lag)
        features[f"close_pct_lag_{lag}"] = close_pct.shift(lag)

    # --- New features based on raw price data ---

    # Volume
    if "Volume" in df.columns:
        features["Volume"] = df["Volume"]
        features["avg_volume_20"] = sma(df["Volume"], 20)

    # Momentum Indicators
    features["RSI"] = rsi_indicator(df["Close"], n=14)
    features["AO"] = awesome_oscillator(df["High"], df["Low"])
    features["WR"] = willr(df["High"], df["Low"], df["Close"])
    features["ROC"] = roc(df["Close"])
    features = pd.concat(
        [features, ultimate_oscillator(df["High"], df["Low"], df["Close"])], axis=1
    )
    features = pd.concat([features, true_strength_index(df["Close"])], axis=1)
    stoch_price = stochastic_oscillator(df["High"], df["Low"], df["Close"])
    features["Stoch_K"] = stoch_price["%K"]
    features["Stoch_D"] = stoch_price["%D"]

    # Trend Indicators
    macd_price_df = macd(df["Close"])
    features["MACD"] = macd_price_df["MACD"]
    features["MACD_Signal"] = macd_price_df["Signal"]
    features["MACD_Hist"] = macd_price_df["Hist"]
    features = pd.concat([features, adx(df["High"], df["Low"], df["Close"])], axis=1)
    features = pd.concat([features, aroon(df["High"], df["Low"])], axis=1)
    features["CCI"] = cci(df["High"], df["Low"], df["Close"])
    features = pd.concat([features, stc(df["Close"])], axis=1)
    vortex_df = vortex(df["High"], df["Low"], df["Close"])
    features = pd.concat([features, vortex_df], axis=1)
    if "VTXP_14" in features.columns and "VTXM_14" in features.columns:
        features["VORTEX_diff"] = features["VTXP_14"] - features["VTXM_14"]

    # Fluctuation Indicators
    bbands = bollinger_bands(df["Close"])
    if bbands is not None and not bbands.empty:
        features["BBP"] = bbands.get("BBP_20_2.0")

    keltner = keltner_channels(df["High"], df["Low"], df["Close"])
    if keltner is not None and not keltner.empty:
        kcu = keltner.get("KCU_20_2.0")
        kcl = keltner.get("KCL_20_2.0")
        if kcu is not None and kcl is not None:
            kc_range = kcu - kcl
            features["KCP"] = (df["Close"] - kcl) / kc_range.replace(0, np.nan)

    donchian = donchian_channels(df["High"], df["Low"])
    if donchian is not None and not donchian.empty:
        dcu = donchian.get("DCU_20_20")
        dcl = donchian.get("DCL_20_20")
        if dcu is not None and dcl is not None:
            dc_range = dcu - dcl
            features["DCP"] = (df["Close"] - dcl) / dc_range.replace(0, np.nan)

    # EMA features
    emas = [10, 15, 20, 30, 40, 50, 60]
    for e in emas:
        features[f"above_ema_{e}"] = (df["Close"] > ewm(df["Close"], span=e)).astype(
            int
        )

    # Consecutive run feature
    signs = np.sign(close_pct)
    signs = signs.replace(0, np.nan).ffill().fillna(0).astype(int)
    blocks = signs.diff().ne(0).cumsum()
    features["run"] = signs.groupby(blocks).cumsum()

    # Fill NaN values that might have been generated
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.bfill(inplace=True)
    features.ffill(inplace=True)

    return features


def create_target_variable(
    df: pd.DataFrame,
    method: str = "ao_on_pct_change",
    peak_distance: int = 1,
    peak_threshold: float = 0,
    std_fraction: float = 1.0,
) -> pd.DataFrame:
    """
    Identifies local tops (1), bottoms (-1), and non-reversal points (0)
    using different methods, and returns the DataFrame with a 'target' column.

    Methods:
    - 'ao_on_price': Awesome Oscillator on actual prices.
    - 'ao_on_pct_change': Awesome Oscillator on price percentage changes.
    - 'pct_change_on_ao': Percentage change of Awesome Oscillator on actual prices.
    - 'pct_change_std': Target based on closing price pct_change exceeding 1 std dev.
    """
    if method == "pct_change_std":
        window = 24 * 7
        close_pct_change = df["Close"].pct_change()
        # The target is based on the NEXT period's price change.
        future_pct_change = close_pct_change.shift(-1)
        rolling_std = close_pct_change.rolling(window=window).std()

        df["target"] = 0
        df.loc[future_pct_change >= (rolling_std * std_fraction), "target"] = 1
        df.loc[future_pct_change <= -(rolling_std * std_fraction), "target"] = -1
        return df

    if method == "ao_on_pct_change":
        # computing the peaks from the awesome oscillator from the pct_change of the values
        high_pct = df["High"].pct_change().fillna(0)
        low_pct = df["Low"].pct_change().fillna(0)
        ao = awesome_oscillator(high_pct, low_pct)
    elif method == "ao_on_price":
        # computing the peaks from the awesome oscillator from the actual price values
        ao = awesome_oscillator(df["High"], df["Low"])
    elif method == "pct_change_on_ao":
        # computing the peaks from the pct_change of the awesome oscillator from the actual price values
        ao_price = awesome_oscillator(df["High"], df["Low"])
        ao = ao_price.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    else:
        raise ValueError(
            f"Invalid method '{method}' specified for create_target_variable"
        )

    if ao is None or ao.isnull().all():
        # If AO can't be calculated, label all points as neutral
        df["target"] = 0
        return df

    # Find peaks (tops) and troughs (bottoms) in the AO
    peaks, _ = find_peaks(ao, distance=peak_distance, threshold=peak_threshold)
    troughs, _ = find_peaks(-ao, distance=peak_distance, threshold=peak_threshold)

    # Create the target column, default to 0 (neutral)
    df["target"] = 0
    df.loc[df.index[peaks], "target"] = 1  # Tops
    df.loc[df.index[troughs], "target"] = -1  # Bottoms

    return df


from functools import partial

import optuna
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score

from backtest_utils import fetch_historical_data


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
