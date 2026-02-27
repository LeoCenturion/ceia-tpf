import pandas as pd
import numpy as np
from src.backtesting.cpcv import (
    partition_data,
    generate_combinatorial_splits,
    get_purged_train_test_split,
    construct_backtest_paths,
)


def dummy_model(test_data: pd.DataFrame) -> pd.DataFrame:
    """A simple dummy model that requires no training."""
    # Example: predict 1 for all samples
    predictions = pd.Series(1, index=test_data.index)
    return predictions


def run_cpcv(
    data: pd.DataFrame, t1: pd.Series, n_groups: int, k: int, embargo_pct: float
):
    """
    Runs the Combinatorial Purged Cross-Validation process.
    """
    groups = partition_data(data, n_groups)
    splits = generate_combinatorial_splits(n_groups, k)

    all_predictions = {i: [] for i in range(n_groups)}

    for train_split, test_split in splits:
        train_indices = pd.concat([groups[i] for i in train_split]).index
        test_indices = pd.concat([groups[i] for i in test_split]).index

        _train_data, test_data = get_purged_train_test_split(
            data, t1, train_indices, test_indices, embargo_pct
        )

        # Here you would train your model and get predictions
        predictions = dummy_model(test_data)

        for group_idx in test_split:
            group_data = groups[group_idx]
            # It's possible that some data from the group was purged, so we only take predictions for the remaining indices
            group_predictions = predictions[predictions.index.isin(group_data.index)]
            all_predictions[group_idx].append(group_predictions)

    return all_predictions


if __name__ == "__main__":
    # Create some dummy data for testing
    data = pd.DataFrame(
        np.random.rand(100, 2),
        columns=["A", "B"],
        index=pd.to_datetime(pd.date_range("2020-01-01", periods=100)),
    )
    t1 = pd.Series(data.index + pd.Timedelta(days=1), index=data.index)

    N_GROUPS = 10
    K_TEST_GROUPS = 2

    all_predictions = run_cpcv(
        data, t1, n_groups=N_GROUPS, k=K_TEST_GROUPS, embargo_pct=0.01
    )
    paths = construct_backtest_paths(
        all_predictions, n_groups=N_GROUPS, k=K_TEST_GROUPS
    )

    print(f"Number of paths constructed: {len(paths)}")
    for i, path in enumerate(paths):
        print(f"Path {i + 1} length: {len(path)}")
