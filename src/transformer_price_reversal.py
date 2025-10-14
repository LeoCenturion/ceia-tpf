import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import matplotlib.pyplot as plt

from backtest_utils import fetch_historical_data


def main():
    """
    Main function to run a Chronos forecast using AutoGluon.
    """
    # 1. Load Data
    print("Loading historical data...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        timeframe= "1h",
        start_date="2022-01-01T00:00:00Z"
    )

    # Prepare data for AutoGluon: requires 'timestamp', 'target', and 'item_id' columns.
    data_ag = data[['Close']].copy()
    data_ag.reset_index(inplace=True)
    data_ag.rename(columns={'index': 'timestamp', 'Close': 'target'}, inplace=True)
    data_ag['item_id'] = 'BTCUSDT'

    # 2. Split data: 70% for training, 30% for validation
    train_end_index = int(len(data_ag) * 0.7)
    train_data = data_ag.iloc[:train_end_index]
    test_data = data_ag.iloc[train_end_index:]

    print(f"Training data from {train_data['timestamp'].min()} to {train_data['timestamp'].max()} ({len(train_data)} points)")
    print(f"Validation data from {test_data['timestamp'].min()} to {test_data['timestamp'].max()} ({len(test_data)} points)")

    train_ts_df = TimeSeriesDataFrame.from_data_frame(
        train_data,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    # The prediction length is the size of our validation set
    prediction_length = len(test_data)

    # 3. Initialize and fit the TimeSeriesPredictor
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        eval_metric="MASE",
        target="target",
        freq="1h"
    )
    print(predictor)
    print("\nFitting Chronos model...")
    predictor.fit(
        train_ts_df,
        hyperparameters={
            "Chronos": {"model_path": "amazon/chronos-bolt-base"},
        },
        time_limit=600  # 10-minute time limit for fitting
    )

    # Print the layers of the fitted model
    try:
        best_model_name = predictor.get_model_best()
        if 'Chronos' in best_model_name:
            trainer = predictor._trainer
            autogluon_model = trainer.load_model(best_model_name)
            # The underlying PyTorch/HF model is stored in the `model` attribute
            pytorch_model = autogluon_model.model
            print("\n--- Chronos Model Layers ---")
            print(pytorch_model)
            print("--------------------------\n")
        else:
            print(f"\nBest model is '{best_model_name}', not Chronos. Cannot print layers.\n")
    except Exception as e:
        print(f"\nCould not print model layers: {e}\n")

    # 4. Generate forecast
    print("\nGenerating forecast...")
    predictions = predictor.predict(train_ts_df)

    # 5. Evaluate and plot results
    # The `score` method evaluates predictions against the ground truth.
    # We pass the entire dataset to `evaluate`, and AutoGluon automatically
    # uses the last `prediction_length` time steps as the validation set.
    full_ts_df = TimeSeriesDataFrame.from_data_frame(
        data_ag,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    evaluation = predictor.evaluate(full_ts_df)
    print("\nForecast Evaluation (on validation data):")
    print(evaluation)


    # Plotting
    plt.figure(figsize=(15, 8))
    
    # Plot training data
    plt.plot(train_data['timestamp'], train_data['target'], label="Training Data", color='blue')
    
    # Plot validation data (ground truth)
    plt.plot(test_data['timestamp'], test_data['target'], label="Actual Values (Validation)", color='green')

    # Plot forecast
    predicted_timestamps = predictions.index.get_level_values('timestamp')
    plt.plot(predicted_timestamps, predictions['mean'], label="Chronos Forecast", linestyle='--', color='red')

    # Plot prediction quantiles
    plt.fill_between(
        predicted_timestamps,
        predictions["0.1"],
        predictions["0.9"],
        color="red",
        alpha=0.1,
        label="10%-90% confidence interval",
    )

    plt.title("BTCUSDT Hourly Price: Chronos Forecast vs. Actual")
    plt.xlabel("Date")
    plt.ylabel("Price (USDT)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    print("\nDisplaying plot. Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
