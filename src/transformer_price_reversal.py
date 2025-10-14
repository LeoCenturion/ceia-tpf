import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, ChronosForCausalLM
import optuna
from functools import partial

from backtest_utils import fetch_historical_data

# Note: The model "amazon/chronos-bolt-base" was not found.
# Assuming a typo and using "amazon/chronos-t5-base" instead.


# --- Part 1: Volume Bar Aggregation (from xgboost_price_reversal_palazzo.py) ---

def aggregate_to_volume_bars(df, volume_threshold=50000):
    """
    Aggregates time-series data into volume bars based on a volume threshold.
    """
    print(f"Aggregating data into volume bars of {volume_threshold} units...")
    bars = []
    current_bar_data = []
    cumulative_volume = 0

    for index, row in df.iterrows():
        current_bar_data.append(row)
        cumulative_volume += row["volume"]
        if cumulative_volume >= volume_threshold:
            bar_df = pd.DataFrame(current_bar_data)

            open_time = bar_df.index[0]
            close_time = bar_df.index[-1]
            open_price = bar_df["Open"].iloc[0]
            close_price = bar_df["close"].iloc[-1]

            bars.append(
                {
                    "open_time": open_time,
                    "close_time": close_time,
                    "open_price": open_price,
                    "close_price": close_price,
                    "total_volume": cumulative_volume,
                }
            )
            current_bar_data = []
            cumulative_volume = 0

    volume_bars_df = pd.DataFrame(bars)

    if volume_bars_df.empty:
        print("Warning: No volume bars created. Threshold might be too high.")
    else:
        print(f"Aggregation complete. {len(volume_bars_df)} volume bars created.\n")
    return volume_bars_df


# --- Part 2: Model Backtesting ---

def manual_backtest_forecast(volume_bars: pd.DataFrame, model, tokenizer, context_length: int, trade_threshold: float, test_size: float = 0.3):
    """
    Performs a manual walk-forward backtest using a forecasting model.
    """
    split_index = int(len(volume_bars) * (1 - test_size))
    test_data = volume_bars.iloc[split_index:]

    if len(test_data) < 2:
        return 0.0, 0.0, 0.0

    capital = 100_000.0
    portfolio_history = [capital]
    is_long = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Starting walk-forward backtest...")
    for i in range(len(test_data) - 1):
        # Calculate PnL for the bar that just closed (from i to i+1)
        if is_long:
            entry_price = test_data['close_price'].iloc[i]
            exit_price = test_data['close_price'].iloc[i + 1]
            capital *= (exit_price / entry_price)
        portfolio_history.append(capital)

        # Decide on position for the *next* bar (from i+1 to i+2)
        # Decision is based on data up to and including bar `i`
        current_data_index = split_index + i
        context_end_idx = current_data_index + 1
        context_start_idx = max(0, context_end_idx - context_length)

        context_series = volume_bars['close_price'].iloc[context_start_idx:context_end_idx]
        context = torch.tensor(context_series.values, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            forecast = model.forecast(context, prediction_length=1)
        
        predicted_price = forecast[0, 0, 0].item()
        current_price = context_series.iloc[-1]

        if predicted_price > current_price * (1 + trade_threshold):
            is_long = True
        else:
            is_long = False
        
        if (i + 1) % 100 == 0 or (i + 1) == len(test_data) - 1:
            print(f"Processed {i+1}/{len(test_data) - 1} steps...")

    # Calculate performance metrics
    total_return = (portfolio_history[-1] / portfolio_history[0]) - 1
    returns = pd.Series(portfolio_history).pct_change().dropna()

    if returns.std() > 0:
        # Simple, non-annualized Sharpe Ratio for comparison
        sharpe_ratio = returns.mean() / returns.std()
    else:
        sharpe_ratio = 0.0
    
    pv_series = pd.Series(portfolio_history)
    peak = pv_series.expanding(min_periods=1).max()
    drawdown = (pv_series / peak) - 1
    max_drawdown = drawdown.min()

    print(f"Backtest complete. Final Return: {total_return:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}")
    return sharpe_ratio, total_return, max_drawdown


# --- Part 3: Optuna Optimization ---

def objective(trial: optuna.Trial, minute_data: pd.DataFrame, model, tokenizer) -> float:
    """
    Optuna objective function to tune hyperparameters for the Chronos strategy.
    """
    # Define hyperparameters
    volume_threshold = trial.suggest_int('volume_threshold', 25000, 100000)
    context_length = trial.suggest_int('context_length', 60, 300, step=30)
    trade_threshold = trial.suggest_float('trade_threshold', 0.0005, 0.01, log=True)
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: {trial.params}")
    
    # Run pipeline
    volume_bars = aggregate_to_volume_bars(minute_data.copy(), volume_threshold=volume_threshold)
    
    if len(volume_bars) < context_length + 100:
        print("Not enough volume bars created for a meaningful backtest. Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    sharpe, total_return, max_dd = manual_backtest_forecast(
        volume_bars,
        model,
        tokenizer,
        context_length=context_length,
        trade_threshold=trade_threshold
    )
    
    trial.set_user_attr("total_return", total_return)
    trial.set_user_attr("max_drawdown", max_dd)

    # Return Sharpe Ratio as the metric to maximize
    return sharpe if not np.isnan(sharpe) else 0.0


def main():
    """
    Main function to run the Optuna hyperparameter optimization study.
    """
    N_TRIALS = 30
    MODEL_NAME = "amazon/chronos-t5-base"

    # 1. Load Model
    print(f"Loading transformer model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ChronosForCausalLM.from_pretrained(MODEL_NAME)
    
    # 2. Load Data
    print("Loading 1-minute historical data...")
    minute_data = fetch_historical_data(
        timeframe="1m",
        data_path='/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv'
    )
    minute_data.rename(columns={'Close': 'close', 'Volume': 'volume'}, inplace=True)
    
    # 3. Setup and Run Optuna Study
    db_file_name = "optuna-study"
    study_name_in_db = 'chronos_forecasting_strategy_v1'
    storage_name = f"sqlite:///{db_file_name}.db"
    print(f"Starting Optuna study: '{study_name_in_db}'. Storage: {storage_name}")

    objective_with_data = partial(objective, minute_data=minute_data, model=model, tokenizer=tokenizer)
    
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name_in_db,
        storage=storage_name,
        load_if_exists=True
    )
    # n_jobs=1 is recommended for GPU-based models to avoid memory conflicts
    study.optimize(objective_with_data, n_trials=N_TRIALS, n_jobs=1)
    
    # 4. Print Study Results
    print("\n--- Optuna Study Best Results ---")
    try:
        best_trial = study.best_trial
        print(f"Best trial value (Sharpe Ratio): {best_trial.value}")
        print("Best parameters found:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print("Associated metrics:")
        print(f"  Total Return: {best_trial.user_attrs.get('total_return', 'N/A'):.2%}")
        print(f"  Max Drawdown: {best_trial.user_attrs.get('max_drawdown', 'N/A'):.2%}")
    except ValueError:
        print("No successful trials were completed.")


if __name__ == "__main__":
    main()
