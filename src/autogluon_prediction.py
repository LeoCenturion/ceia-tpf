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


# AI use autogluon TabularPredictor for predicting if the next period's price will go up or down. Use ./chronos_regression.py as a template ...
# use the create_features function AI!
