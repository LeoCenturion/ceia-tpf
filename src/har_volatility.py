import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta
import os
import sys

# Add the parent directory of src to sys.path to allow importing from backtest_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest_utils import fetch_historical_data


def calculate_realized_volatility(df: pd.DataFrame, freq: str) -> pd.Series:
    """
    Calculates realized variance (RV) for a given frequency from high-frequency data.

    Args:
        df: DataFrame with 'close' prices indexed by timestamp.
        freq: The frequency to resample to (e.g., 'D' for daily).
              This determines what constitutes 'RV_t_daily' in the HAR model.
    Returns:
        A Series of realized variance values.
    """
    # Calculate log returns
    df["log_return"] = np.log(
        df["Close"]
    ).diff()  # Use 'Close' as per fetch_historical_data output

    # Aggregate squared log returns to the desired frequency
    # Assuming 'Close' price is available for each 5-minute bar
    realized_variance = df["log_return"].pow(2).resample(freq).sum()

    return realized_variance.dropna()


def build_har_model_data(daily_rv: pd.Series) -> pd.DataFrame:
    """
    Prepares data for the HAR model by creating lagged daily, weekly, and monthly RV components.

    Args:
        daily_rv: A Series of daily realized variances.
    Returns:
        DataFrame with dependent and independent variables for the HAR model.
    """
    # Lagged daily RV
    df = pd.DataFrame(daily_rv)
    df.columns = ["RV_daily"]
    df["RV_daily_lag"] = df["RV_daily"].shift(1)

    # Lagged weekly RV (average of past 5 daily RVs)
    df["RV_weekly_lag"] = df["RV_daily"].shift(1).rolling(window=5).mean()

    # Lagged monthly RV (average of past 22 daily RVs)
    df["RV_monthly_lag"] = df["RV_daily"].shift(1).rolling(window=22).mean()

    # Dependent variable (next day's RV)
    df["RV_daily_next"] = df["RV_daily"].shift(-1)

    df = df.dropna()
    return df


def har_forecast(train_data: pd.DataFrame) -> float:
    """
    Fits a HAR model to training data and provides a one-step-ahead forecast of RV.

    Args:
        train_data: DataFrame with HAR model variables (RV_daily_next, RV_daily_lag, RV_weekly_lag, RV_monthly_lag).
    Returns:
        The one-step-ahead forecast of realized variance.
    """
    if train_data.empty:
        raise ValueError("Training data is empty, cannot fit HAR model.")

    # Define dependent and independent variables
    y = train_data["RV_daily_next"]
    X = train_data[["RV_daily_lag", "RV_weekly_lag", "RV_monthly_lag"]]
    X = sm.add_constant(X)  # Add an intercept

    # Fit OLS model
    model = sm.OLS(y, X, missing="drop")
    results = model.fit()

    # Prepare data for one-step-ahead forecast (using the last available lagged values)
    last_lags = train_data[["RV_daily_lag", "RV_weekly_lag", "RV_monthly_lag"]].iloc[-1]
    forecast_X = sm.add_constant(pd.DataFrame([last_lags]))

    # Get forecast
    forecast = results.predict(forecast_X)[0]
    return forecast


def backtest_har_volatility(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    start_date: str = "2024-01-01",
    end_date: str = "2025-01-01",
    initial_train_period_days: int = 90,
    forecast_freq: str = "D",  # Forecast daily RV
) -> pd.DataFrame:
    """
    Backtests the HAR volatility forecasting model using an expanding window.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT").
        timeframe: Data timeframe (e.g., "5m").
        start_date: Start date for historical data fetching.
        end_date: End date for historical data fetching.
        initial_train_period_days: Initial number of days for the training set.
        forecast_freq: The frequency of the realized volatility being forecast (e.g., 'D' for daily).
    Returns:
        DataFrame containing actual and forecasted realized volatility.
    """
    print(
        f"Fetching {symbol} {timeframe} historical data from {start_date} to {end_date}..."
    )
    # fetch_historical_data from backtest_utils returns pd.DataFrame
    # Ensure it's sorted by timestamp if not already
    raw_data = fetch_historical_data(symbol, timeframe, start_date, end_date)

    if raw_data.empty or "close" not in raw_data.columns:
        raise ValueError(
            "Failed to fetch historical data or 'close' column is missing."
        )

    # Ensure index is datetime for resampling
    raw_data.reset_index(inplace=True)  # Reset index to access 'timestamp' as a column
    raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])
    raw_data.set_index("timestamp", inplace=True)
    raw_data = raw_data.sort_index()

    print("Calculating daily realized volatility...")
    daily_rv = calculate_realized_volatility(raw_data, forecast_freq)
    har_df = build_har_model_data(daily_rv)

    # Set up backtesting dates
    # Assuming daily_rv index is datetime objects
    all_dates = har_df.index

    # Minimum training data length required for HAR (22 days for monthly RV + 1 lag)
    min_train_size = 23  # 22 for monthly mean + 1 for lag
    if len(all_dates) < min_train_size:
        raise ValueError(
            f"Not enough daily RV data ({len(all_dates)} days) for initial HAR model training. Need at least {min_train_size} days."
        )

    # Find the starting point for the backtest
    # The first date where we have enough historical data for the HAR features
    first_forecast_date_idx = 0
    while first_forecast_date_idx < len(all_dates):
        current_date = all_dates[first_forecast_date_idx]
        # Check if we have enough data up to current_date for the initial train period
        # and for the HAR model's lags (22 days for monthly RV)
        if current_date - all_dates[0] >= timedelta(days=initial_train_period_days):
            break
        first_forecast_date_idx += 1

    if first_forecast_date_idx >= len(all_dates):
        raise ValueError(
            "Not enough data to form initial training period for backtest."
        )

    results = []

    print("Starting HAR model backtest...")
    # Loop through the dates for forecasting
    for i in range(first_forecast_date_idx, len(all_dates)):
        current_forecast_date = all_dates[i]

        # Expanding window: train on all data up to the current forecast date's lags
        train_window_end_date = current_forecast_date - timedelta(
            days=1
        )  # Train on data *before* the forecast day

        # Select data for training, ensuring we have enough historical context for HAR lags
        train_har_df = har_df.loc[har_df.index <= train_window_end_date]

        if len(train_har_df) < min_train_size:
            print(
                f"Skipping {current_forecast_date}: not enough data for HAR training ({len(train_har_df)}/{min_train_size})"
            )
            continue

        try:
            forecast_rv = har_forecast(train_har_df)
            actual_rv = har_df.loc[
                current_forecast_date, "RV_daily"
            ]  # Actual RV for the day being forecast

            results.append(
                {
                    "date": current_forecast_date,
                    "actual_rv": actual_rv,
                    "forecast_rv": forecast_rv,
                }
            )
            print(
                f"Date: {current_forecast_date.strftime('%Y-%m-%d')}, Actual RV: {actual_rv:.8f}, Forecast RV: {forecast_rv:.8f}"
            )
        except Exception as e:
            print(
                f"Error forecasting for {current_forecast_date.strftime('%Y-%m-%d')}: {e}"
            )
            results.append(
                {
                    "date": current_forecast_date,
                    "actual_rv": har_df.loc[current_forecast_date, "RV_daily"],
                    "forecast_rv": np.nan,  # Mark failed forecasts
                }
            )

    results_df = pd.DataFrame(results).set_index("date")
    return results_df


if __name__ == "__main__":
    print("Running HAR Volatility Backtest...")
    # Define backtest parameters
    symbol = "BTCUSDT"
    timeframe = "5m"
    start_date = "2024-01-01"
    end_date = "2025-01-01"  # Adjust as needed
    initial_train_period_days = 90  # Initial training for 3 months

    backtest_results = backtest_har_volatility(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_train_period_days=initial_train_period_days,
    )

    if not backtest_results.empty:
        print("\nBacktest Results:")
        print(backtest_results.head())
        print("\nEvaluation Metrics:")
        # Evaluate performance (e.g., RMSE)
        rmse = np.sqrt(
            (
                (backtest_results["actual_rv"] - backtest_results["forecast_rv"]) ** 2
            ).mean()
        )
        print(f"Root Mean Squared Error (RMSE): {rmse:.8f}")

        # You might want to save these results
        results_path = "har_backtest_results.csv"
        backtest_results.to_csv(results_path)
        print(f"Backtest results saved to {results_path}")
    else:
        print("No backtest results generated.")
