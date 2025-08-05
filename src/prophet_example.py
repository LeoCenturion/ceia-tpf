import numpy as np
import pandas as pd
from prophet import Prophet

from src.backtest import TradingBacktester


def prophet_expanding_window_validation(
    data: pd.DataFrame, min_train_size: int = 365 * 24, step: int = 30 * 24
):
    """
    Performs expanding window cross-validation for a Prophet model to assess its predictive accuracy.

    This function simulates a real-world scenario where the model is retrained periodically
    with new data. It iterates through the dataset, using an expanding window for training
    and predicting the next step.

    :param data: DataFrame with historical price data. Must include 'close' prices and have a DatetimeIndex.
    :param min_train_size: The minimum number of observations in the initial training set. Default is 1 year of hourly data.
    :param step: The number of observations to add to the training set for each new fold. Default is 1 month of hourly data.
    """
    # Prepare data for Prophet
    prophet_df = data.reset_index()[["timestamp", "close"]].rename(
        columns={"timestamp": "ds", "close": "y"}
    )
    prophet_df["y"] = prophet_df["y"].pct_change()
    prophet_df = prophet_df.dropna()

    correct_predictions = 0
    total_predictions = 0

    accuracies = []

    print("Starting Prophet expanding window validation...")
    for i in range(min_train_size, len(prophet_df), step):
        train_df = prophet_df.iloc[:i]
        test_df = prophet_df.iloc[i : i + step]

        if test_df.empty:
            continue

        model = Prophet()
        model.fit(train_df)

        # Predict for the test period
        future = test_df[["ds"]]
        forecast = model.predict(future)

        # Compare predicted direction with actual direction
        merged_df = test_df.merge(forecast, on="ds")

        # Predicted signal: 1 if trend is positive, -1 if negative
        merged_df["predicted_signal"] = np.sign(merged_df["yhat"])
        # Actual signal: 1 if price change is positive, -1 if negative
        merged_df["actual_signal"] = np.sign(merged_df["y"])

        # Correct prediction if signals match
        correct_in_fold = (
            merged_df["predicted_signal"] == merged_df["actual_signal"]
        ).sum()
        total_in_fold = len(merged_df)

        correct_predictions += correct_in_fold
        total_predictions += total_in_fold

        if total_in_fold > 0:
            accuracy = correct_in_fold / total_in_fold
            accuracies.append(accuracy)
            print(
                f"Fold with training data until {train_df['ds'].iloc[-1]}: Accuracy = {accuracy:.2%}"
            )

    if total_predictions > 0:
        overall_accuracy = correct_predictions / total_predictions
        print(
            f"\nOverall Prophet model accuracy (expanding window): {overall_accuracy:.2%}"
        )
    else:
        print("Not enough data to perform expanding window validation.")

    return accuracies


def main():
    """
    Main function to run the Prophet expanding window validation example.
    """
    # Initialize backtester to fetch data
    backtester = TradingBacktester()

    # Fetch historical data
    historical_data = backtester.fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2018-01-01",
        end_date="2024-01-01",
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
    )

    print("\n--- Prophet Expanding Window Validation ---")
    prophet_expanding_window_validation(
        historical_data, min_train_size=365 * 24, step=30 * 24
    )


if __name__ == "__main__":
    main()
