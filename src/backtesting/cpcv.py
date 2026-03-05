import logging
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def partition_data(data: pd.DataFrame, n_groups: int) -> list:
    """Partitions the data into N chronological groups."""
    group_size = len(data) // n_groups
    groups = []
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size if i < n_groups - 1 else len(data)
        groups.append(data.iloc[start:end])
    return groups


def generate_combinatorial_splits(n_groups: int, k: int) -> list:
    """Generates all combinations of training and testing splits."""
    all_indices = list(range(n_groups))
    test_splits = list(combinations(all_indices, k))
    train_splits = [
        tuple(sorted(list(set(all_indices) - set(test)))) for test in test_splits
    ]
    return list(zip(train_splits, test_splits))


def get_purged_train_test_split(
    data: pd.DataFrame,
    t1: pd.Series,
    train_indices: list,
    test_indices: list,
    embargo_pct: float,
):
    """
    Applies purging and embargoing to a train-test split.

    Args:
        data (pd.DataFrame): The full dataset.
        t1 (pd.Series): A series with the end time of each event.
        train_indices (list): The indices of the training data.
        test_indices (list): The indices of the testing data.
        embargo_pct (float): The percentage of the test set size to use for embargoing.

    Returns:
        tuple: Purged training data and testing data.
    """
    train_data = data.loc[train_indices]
    test_data = data.loc[test_indices]

    # Purging
    # if t1 is not None:
    #     test_times = t1[test_data.index]
    #     train_times = t1[train_data.index]

    #     # Events in train_data that overlap with test_data
    #     overlapping_events = train_times[
    #         (train_times.index >= test_times.index.min())
    #         & (train_times.index <= test_times.index.max())
    #     ]

    #     # Purge train_data
    #     train_data = train_data.drop(index=overlapping_events.index)

    # Embargoing
    # embargo_size = int(len(test_data) * embargo_pct)
    # if embargo_size > 0:
    #     last_test_time = test_data.index.max()
    #     embargo_start_time = last_test_time + pd.Timedelta(
    #         seconds=1
    #     )  # Start embargo right after the test set
    #     embargo_end_time = (
    #         embargo_start_time + pd.DateOffset(days=embargo_size)
    #     )  # Approximate embargo period

    #     # Drop training samples within embargo period
    #     train_data = train_data[train_data.index > embargo_end_time]

    return train_data, test_data


def get_num_paths(N, k):
    """Calculate the number of backtest paths."""
    if N == 0:
        return 0
    return int(k / N * comb(N, k))


def construct_backtest_paths(all_predictions: dict, n_groups: int, k: int):
    """
    Stitches together the out-of-sample predictions to form complete backtest paths.
    """
    num_paths = get_num_paths(n_groups, k)
    if num_paths == 0:
        logger.debug("No complete paths can be constructed.")
        return []

    paths = []
    for i in range(num_paths):
        current_path_preds = []
        for g in range(n_groups):
            if i < len(all_predictions[g]):
                current_path_preds.append(all_predictions[g][i])
            else:
                # This should not happen if the math is correct
                logger.debug(f"Warning: Not enough predictions for group {g} to form path {i}.")
                continue

        # Concatenate and sort by time index to form a complete path
        if current_path_preds:
            full_path = pd.concat(current_path_preds).sort_index()
            paths.append(full_path)

    return paths
