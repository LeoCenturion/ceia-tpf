import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
# AI adapt this file like xgboost_price_reversal.py AI!

# --- Part 1: Data Simulation and Volume Bar Creation ---
# The paper uses high-frequency data to construct volume bars.
# We'll simulate 1-minute data to demonstrate the process.


def create_mock_data(days=100):
    """Creates a mock DataFrame of 1-minute BTC price data."""
    print("Step 1: Creating mock high-frequency (1-minute) data...")
    date_rng = pd.to_datetime(
        pd.date_range(start="2022-01-01", periods=days * 24 * 60, freq="T")
    )
    data = pd.DataFrame(date_rng, columns=["timestamp"])

    # Simulate a random walk for price
    price_movements = np.random.randn(len(data)) / 1000 + 0.00001
    data["close"] = 20000 * (1 + price_movements).cumprod()

    # Simulate volume, with some periods of higher activity
    data["volume"] = np.random.randint(5, 50, size=len(data))
    high_activity_spikes = np.random.choice(
        data.index, size=int(len(data) * 0.1), replace=False
    )
    data.loc[high_activity_spikes, "volume"] *= 5  # Make some periods more active

    data.set_index("timestamp", inplace=True)
    print("Mock data created.\n")
    return data


def aggregate_to_volume_bars(df, volume_threshold=50000):
    """
    Aggregates time-series data into volume bars based on a volume threshold.
    This follows the core concept of the dissertation (Section 3.3).
    """
    print(f"Step 2: Aggregating data into volume bars of {volume_threshold} units...")
    bars = []
    current_bar_data = []
    cumulative_volume = 0

    for index, row in df.iterrows():
        current_bar_data.append(row)
        cumulative_volume += row["volume"]

        if cumulative_volume >= volume_threshold:
            bar_df = pd.DataFrame(current_bar_data)

            # Bar characteristics
            open_time = bar_df.index[0]
            close_time = bar_df.index[-1]
            open_price = bar_df["close"].iloc[0]
            close_price = bar_df["close"].iloc[-1]

            # Calculate intra-bar volatility for labeling (σv)
            # The paper uses log-returns for some calculations.
            bar_log_returns = np.log(
                bar_df["close"] / bar_df["close"].shift(1)
            ).dropna()
            intra_bar_std = bar_log_returns.std()

            bars.append(
                {
                    "open_time": open_time,
                    "close_time": close_time,
                    "open_price": open_price,
                    "close_price": close_price,
                    "total_volume": cumulative_volume,
                    "intra_bar_std": intra_bar_std if len(bar_log_returns) > 1 else 0,
                }
            )

            # Reset for the next bar
            current_bar_data = []
            cumulative_volume = 0

    volume_bars_df = pd.DataFrame(bars)
    # Calculate bar returns (rv), which will be used for labeling
    volume_bars_df["bar_return"] = (
        volume_bars_df["close_price"] / volume_bars_df["open_price"]
    ) - 1

    print(f"Aggregation complete. {len(volume_bars_df)} volume bars created.\n")
    return volume_bars_df


# --- Part 2: Target Labeling and Feature Engineering ---


def create_labels(df, tau=0.35):
    """
    Creates target labels based on the triple-barrier method variation
    described in Section 3.3.2, Equation 3.1.
    """
    print("Step 3: Creating target labels...")
    df["label"] = 0
    # Shift returns and volatility to use future information for labeling ONLY
    df["next_bar_return"] = df["bar_return"].shift(-1)

    # Conditions for label = 1 ('top')
    # Condition 1: Next bar's return must be positive.
    cond1 = df["next_bar_return"] >= 0
    # Condition 2: Next bar's return must exceed current return + volatility threshold.
    cond2 = df["next_bar_return"] >= (df["bar_return"] + df["intra_bar_std"] * tau)

    df.loc[cond1 & cond2, "label"] = 1

    # Clean up columns used only for labeling
    df.dropna(subset=["next_bar_return"], inplace=True)
    df.drop(columns=["next_bar_return"], inplace=True)

    print(
        f"Labeling complete. Class distribution:\n{df['label'].value_counts(normalize=True)}\n"
    )
    return df


def create_features(df):
    """
    Creates simple features for the model. The paper uses 796 features;
    we will simulate a few for demonstration (Section 3.3.3).
    """
    print("Step 4: Creating features...")
    df["feature_return_lag_1"] = df["bar_return"].shift(1)
    df["feature_volatility_lag_1"] = df["intra_bar_std"].shift(1)
    df["feature_rolling_mean_return_5"] = (
        df["bar_return"].shift(1).rolling(window=5).mean()
    )
    df["feature_rolling_std_return_5"] = (
        df["bar_return"].shift(1).rolling(window=5).std()
    )

    # Drop rows with NaNs created by lagging/rolling features
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("Features created.\n")
    return df


# --- Part 3: Model Training and Evaluation ---


def train_and_evaluate_xgboost(df):
    """
    Trains an XGBoost classifier and evaluates its performance.
    """
    print("Step 5: Training and Evaluating XGBoost Model...")
    features = [col for col in df.columns if "feature_" in col]
    target = "label"

    X = df[features]
    y = df[target]

    # Split data into training and testing sets (70/30 split as in paper)
    split_index = int(len(X) * 0.7)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Scale features using MinMaxScaler as per the paper (Section 3.3.3)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # The paper uses GridSearchCV. For simplicity, we'll use the final optimized parameters
    # from Table 4.8 in the dissertation.
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.05,  # A slightly more conservative rate than 0.55
        "gamma": 0.92,
        "reg_lambda": 3.73,
        "min_child_weight": 1.39,
        "use_label_encoder": False,
        "seed": 42,
    }

    # Handle class imbalance for training data as mentioned in the paper
    # The paper undersamples; an easier approach is to use scale_pos_weight
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    model = xgb.XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight)

    model.fit(X_train_scaled, y_train, verbose=False)

    # Make predictions on the test set
    predictions = model.predict(X_test_scaled)

    print("Model Evaluation on Test Set:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions))

    # Save the test set and predictions for the backtest
    test_df = df.iloc[split_index:].copy()
    test_df["prediction"] = predictions

    print("Training and evaluation complete.\n")
    return model, test_df


# --- Part 4: Backtesting the Strategy ---


def run_backtest(test_df):
    """
    Runs a simple backtest based on the model's predictions.
    The strategy is to buy on a predicted '1' and sell on the next bar's close.
    (Section 3.5 Portfolio construction)
    """
    print("Step 6: Running backtest on the trading strategy...")
    initial_capital = 100000
    capital = initial_capital
    portfolio_history = []

    # We need the next bar's close price to calculate profit, so we add it here
    test_df["next_close_price"] = test_df["close_price"].shift(-1)

    for i, row in test_df.iterrows():
        # A signal of 1 means we buy at the current bar's close and sell at the next bar's close
        if row["prediction"] == 1 and not pd.isna(row["next_close_price"]):
            buy_price = row["close_price"]
            sell_price = row["next_close_price"]

            # Simple return calculation
            trade_return = (sell_price / buy_price) - 1
            capital *= 1 + trade_return

        portfolio_history.append(capital)

    # Calculate performance metrics
    final_portfolio_value = portfolio_history[-1]
    total_return = (final_portfolio_value / initial_capital - 1) * 100

    returns_series = (
        pd.Series(portfolio_history, index=test_df.index).pct_change().dropna()
    )
    # Assuming daily bars for annualization factor, adjust if frequency is different
    # Since bar duration varies, a precise annualization is complex. We'll use a common sqrt(252).
    sharpe_ratio = (
        (returns_series.mean() / returns_series.std()) * np.sqrt(252)
        if returns_series.std() > 0
        else 0
    )

    print("Backtest Results:")
    print(f"Initial Portfolio Value: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value:   ${final_portfolio_value:,.2f}")
    print(f"Total Cumulative Return: {total_return:.2f}%")
    print(f"Sharpe Ratio (approx.):  {sharpe_ratio:.2f}")


# --- Main Execution Workflow ---
if __name__ == "__main__":
    # 1. Simulate raw data
    raw_minute_data = create_mock_data(days=150)

    # 2. Aggregate into volume bars
    volume_bars = aggregate_to_volume_bars(raw_minute_data, volume_threshold=50000)

    # 3. Create target labels
    labeled_bars = create_labels(volume_bars, tau=0.35)

    # 4. Engineer features
    final_df = create_features(labeled_bars)

    # 5. Train model
    trained_model, test_results_df = train_and_evaluate_xgboost(final_df)

    # 6. Run backtest
    run_backtest(test_results_df)
