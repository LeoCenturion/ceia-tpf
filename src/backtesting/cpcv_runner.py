import mlflow
import pandas as pd
import numpy as np

from src.backtesting.cpcv import (
    partition_data,
    generate_combinatorial_splits,
    get_purged_train_test_split,
    construct_backtest_paths,
)
from src.backtesting.backtesting import TrialStrategy
from src.backtesting.strategies.statistical_strategies import SmaCross
from src.data_analysis.data_analysis import fetch_historical_data, adjust_data_to_ubtc
from src.modeling.mlflow_utils import MLflowLogger
from src.modeling.chronos_metalabeling_pipeline import ChronosMetaLabelingPipeline

def run_cpcv_for_strategy(
    data: pd.DataFrame,
    t1: pd.Series,
    strategy_class: TrialStrategy,
    strategy_params: dict,
    n_groups: int,
    k_test_groups: int,
    embargo_pct: float,
    experiment_name: str
):
    """
    Runs the full CPCV process for a given strategy and logs results to MLflow.
    """
    logger = MLflowLogger(experiment_name=experiment_name)
    logger.start_run(run_name=f"CPCV_{strategy_class.__name__}")
    
    try:
        # Log main CPCV parameters
        logger.log_params({
            "strategy": strategy_class.__name__,
            "n_groups": n_groups,
            "k_test_groups": k_test_groups,
            "embargo_pct": embargo_pct,
        })
        logger.log_params(strategy_params)

        # --- Step 4: Generate OOS predictions for all splits ---
        groups = partition_data(data, n_groups)
        splits = generate_combinatorial_splits(n_groups, k_test_groups)
        all_predictions = {i: [] for i in range(n_groups)}

        for train_split, test_split in splits:
            train_indices = pd.concat([groups[i] for i in train_split]).index
            test_indices = pd.concat([groups[i] for i in test_split]).index

            _train_data, test_data = get_purged_train_test_split(
                data, t1, train_indices, test_indices, embargo_pct
            )
            
            # If using ChronosMetaLabelingCPCVStrategy, train the pipeline here
            if strategy_class.__name__ == "ChronosMetaLabelingCPCVStrategy":
                try:
                    pipeline_config = strategy_params.get("pipeline_config", {})
                    current_pipeline = ChronosMetaLabelingPipeline(config=pipeline_config)
                    
                    train_df = _train_data
                    
                    # Fit the pipeline on the current training data
                    current_pipeline.fit(train_df, model_cls=strategy_params['model_cls'], model_params=strategy_params['model_params'])
                    
                    # Pass the trained pipeline instance to the strategy
                    strategy_instance = strategy_class(
                        broker=None, 
                        data=test_data, 
                        params={**strategy_params, "trained_pipeline": current_pipeline}
                    )
                except ValueError as e:
                    print(f"Skipping split due to error during pipeline fitting: {e}")
                    continue # Skip to the next split
            else:
                # Existing logic for other strategies
                strategy_instance = strategy_class(broker=None, data=test_data, params=strategy_params)

            predictions = strategy_instance.predict(test_data)

            for group_idx in test_split:
                group_data = groups[group_idx]
                group_predictions = predictions[predictions.index.isin(group_data.index)]
                all_predictions[group_idx].append(group_predictions)

        # --- Step 5: Construct backtest paths ---
        paths = construct_backtest_paths(all_predictions, n_groups, k_test_groups)
        
        print(f"Constructed {len(paths)} backtest paths.")

        # --- Step 6: Evaluate each path and log to MLflow ---
        path_sharpe_ratios = []
        for i, path_predictions in enumerate(paths):
            with mlflow.start_run(run_name=f"path_{i+1}", nested=True):
                # Here, you would calculate the performance of the path.
                # For example, calculate returns based on the signals
                # and then compute the Sharpe ratio.
                
                # Dummy calculation for now:
                path_returns = data['Close'].pct_change().loc[path_predictions.index] * path_predictions
                path_sharpe = path_returns.mean() / path_returns.std() * np.sqrt(365*24) # Annualized Sharpe for hourly data
                if np.isinf(path_sharpe) or np.isnan(path_sharpe):
                    path_sharpe = 0
                path_sharpe_ratios.append(path_sharpe)
                
                mlflow.log_metric("sharpe_ratio", path_sharpe)

        # Log the distribution of performance to the parent run
        if path_sharpe_ratios:
            logger.log_metrics({"sharpe_mean": np.nanmean(path_sharpe_ratios)})
            logger.log_metrics({"sharpe_std": np.nanstd(path_sharpe_ratios)})
            # You can calculate and log PBO here as well
    
    finally:
        logger.end_run()

    return paths

if __name__ == '__main__':
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv"
    data = fetch_historical_data(data_path=data_path, start_date="2022-01-01T00:00:00Z", timeframe="1h")
    data = adjust_data_to_ubtc(data)
    
    # For CPCV, we need t1, which indicates the end of an event.
    # For a simple time-bar based strategy, we can assume the event ends at the next bar.
    t1 = pd.Series(data.index[1:], index=data.index[:-1])
    # For the last element, we can't know the end, so we can drop it or approximate.
    data = data.iloc[:-1]

    run_cpcv_for_strategy(
        data=data,
        t1=t1,
        strategy_class=SmaCross,
        strategy_params={'n1': 10, 'n2': 25},
        n_groups=10,
        k_test_groups=2,
        embargo_pct=0.01,
        experiment_name="CPCV_Backtest"
    )
