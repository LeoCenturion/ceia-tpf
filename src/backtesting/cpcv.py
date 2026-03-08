import logging
import random
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def time_based_partition(index: pd.DatetimeIndex, n_groups: int) -> list:
    """Partitions data into N groups based on time."""
    start_time, end_time = index.min(), index.max()
    path_duration = (end_time - start_time) / n_groups
    time_splits = [start_time + i * path_duration for i in range(n_groups + 1)]
    time_splits[-1] = end_time
    path_indices = [
        np.where((index >= time_splits[i]) & (index < time_splits[i + 1]))[0]
        if i < n_groups - 1
        else np.where((index >= time_splits[i]) & (index <= time_splits[i + 1]))[0]
        for i in range(n_groups)
    ]
    return path_indices


def generate_combinatorial_splits(n_groups: int, k: int) -> list:
    """Generates all combinations of training and testing splits."""
    if k >= n_groups:
        raise ValueError("k must be smaller than n_groups for combinatorial splits.")
    all_indices = list(range(n_groups))
    test_splits = list(combinations(all_indices, k))
    train_splits = [
        tuple(sorted(list(set(all_indices) - set(test)))) for test in test_splits
    ]
    return list(zip(train_splits, test_splits))


def purge_and_embargo_split(
    X: pd.DataFrame,
    t1: pd.Series,
    path_indices: list,
    train_group_idxs: tuple,
    test_group_idxs: tuple,
    pct_embargo: float,
):
    """Applies purging and embargoing to a train-test split."""
    train_indices_orig = np.concatenate(
        [path_indices[i] for i in train_group_idxs if path_indices[i].size > 0]
    )
    test_indices = np.concatenate(
        [path_indices[i] for i in test_group_idxs if path_indices[i].size > 0]
    )

    if test_indices.size == 0 or train_indices_orig.size == 0:
        return np.array([]), np.array([])

    test_path_time_ranges = [
        (X.index[path_indices[i][0]], X.index[path_indices[i][-1]])
        for i in test_group_idxs
        if path_indices[i].size > 0
    ]
    train_times = X.index[train_indices_orig]
    train_t1 = t1.loc[train_times]
    logger.debug(f'Train before purge: {len(train_times)}')
    # Purging
    purge_mask = pd.Series(False, index=train_times)
    for start, end in test_path_time_ranges:
        purge_mask |= (train_times >= start) & (train_times <= end)
        purge_mask |= (train_t1 >= start) & (train_t1 <= end)
    train_indices_purged = train_indices_orig[~purge_mask.values]

    # Embargo
    embargo_td = (X.index[-1] - X.index[0]) * pct_embargo
    train_indices_final = train_indices_purged
    if embargo_td.total_seconds() > 0 and train_indices_purged.size > 0:
        embargo_mask = pd.Series(False, index=X.index[train_indices_purged])
        for _, end in test_path_time_ranges:
            embargo_mask |= (X.index[train_indices_purged] > end) & (
                X.index[train_indices_purged] <= end + embargo_td
            )
        train_indices_final = train_indices_purged[~embargo_mask.values]

    return train_indices_final, test_indices


def _find_paths(splits, n_groups):
    """
    Finds all unique sets of splits that form a complete partition of the N groups.
    This is a recursive backtracking algorithm.
    """
    memo = {}

    def solve(groups_tuple, available_splits_indices):
        if not groups_tuple:
            return [[]]
        groups_tuple = tuple(sorted(groups_tuple))
        state = (groups_tuple, available_splits_indices)
        if state in memo:
            return memo[state]

        res = []
        first_group = groups_tuple[0]

        for i in available_splits_indices:
            split = splits[i]
            if first_group in split:
                remaining_groups = tuple(g for g in groups_tuple if g not in split)

                # New available splits are those that do not overlap with the current split
                new_available_splits_indices = tuple(
                    j
                    for j in available_splits_indices
                    if not set(splits[j]).intersection(split)
                )

                sub_partitions = solve(remaining_groups, new_available_splits_indices)
                for p in sub_partitions:
                    res.append([split] + p)

        memo[state] = res
        return res

    all_splits_indices = tuple(range(len(splits)))
    all_groups = tuple(range(n_groups))
    raw_paths = solve(all_groups, all_splits_indices)

    # Deduplicate paths (the solver might find the same path with splits in a different order)
    unique_paths = set()
    for p in raw_paths:
        canonical_path = tuple(sorted(p))
        unique_paths.add(canonical_path)

    return [list(p) for p in unique_paths]


def construct_backtest_paths(
    split_predictions: list, n_groups: int, k_test_groups: int
):
    """
    Stitches together OOS predictions to form complete backtest paths
    based on the method described by Lopez de Prado, ensuring each
    prediction is used in at most one path.
    """
    all_preds = {p["test_path_idxs"]: p for p in split_predictions}
    all_splits = list(all_preds.keys())

    # Step 1: Find all possible ways to form a complete path (a partition of N groups)
    all_possible_paths = _find_paths(all_splits, n_groups)

    # Step 2: Use a randomized greedy heuristic to select disjoint paths.
    # This is a heuristic for the set-packing problem, which is NP-hard.
    # We try multiple random orderings to find a better packing.
    best_path_selection = []
    for _ in range(20):  # Number of random trials
        random.shuffle(all_possible_paths)

        current_selection = []
        used_splits = set()
        for path_candidate in all_possible_paths:
            candidate_splits = set(path_candidate)
            if used_splits.isdisjoint(candidate_splits):
                current_selection.append(path_candidate)
                used_splits.update(candidate_splits)

        if len(current_selection) > len(best_path_selection):
            best_path_selection = current_selection

    selected_paths = best_path_selection

    # Lopez de Prado's framework suggests a specific number of paths can be formed.
    expected_num_paths = (
        comb(n_groups - 1, k_test_groups - 1)
        if k_test_groups > 0 and n_groups >= k_test_groups
        else 0
    )

    logger.info(
        f"Constructed {len(selected_paths)} unique, disjoint backtest paths."
    )
    if len(selected_paths) < expected_num_paths:
        logger.warning(
            f"Could only construct {len(selected_paths)} paths, "
            f"less than the expected {expected_num_paths}. "
            "Some predictions will not be used."
        )

    # Step 3: Assemble the results for the selected paths
    path_results = []
    for path in selected_paths:
        path_y_true, path_y_pred = [], []
        for split_groups in path:
            if split_groups in all_preds:
                split_results = all_preds[split_groups]
                path_y_true.append(split_results["y_test"])
                path_y_pred.append(split_results["preds"])

        if not path_y_true:
            continue

        path_y_true = np.concatenate(path_y_true)
        path_y_pred = np.concatenate(path_y_pred)

        path_results.append({"y_true": path_y_true, "y_pred": path_y_pred})

    return path_results
