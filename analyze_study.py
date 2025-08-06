import sqlite3
import pandas as pd

# --- Configuration ---
DB_FILE = "ma_crossover_study.db"

# --- Load Data ---
try:
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_FILE)

    # Load the trials table into a pandas DataFrame
    trials_df = pd.read_sql_query("SELECT * FROM trials", conn)
    params_df = pd.read_sql_query("SELECT * FROM trial_params", conn)
    conn.close()

    # Merge trial data with parameters
    df = pd.merge(trials_df, params_df, on="trial_id")

    # Drop trials where the objective value is NaN, which can happen with Sharpe Ratio
    # when no trades are made.
    df.dropna(subset=["value"], inplace=True)

    if df.empty:
        print("Could not find any completed trials with valid objective values.")
        print(
            "This might happen if all trials resulted in an undefined metric (e.g., NaN Sharpe Ratio)."
        )
        exit()

    # Pivot parameters into columns
    df = df.pivot_table(
        index=["trial_id", "value", "state"], columns="param_name", values="param_value"
    ).reset_index()


    # --- Analysis ---
    print("--- Optimization Trials Analysis ---")
    print(f"Loaded {len(df)} trials from '{DB_FILE}'")

    # Display the top 5 best trials based on the 'value'
    print("\n--- Top 5 Best Trials (by Objective Value) ---")
    best_trials = df.sort_values("value", ascending=False).head(5)
    print(best_trials[["value", "short_window", "long_window"]])

except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Please ensure '{DB_FILE}' exists in the correct path and is a valid Optuna study database.")
