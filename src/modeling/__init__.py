import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold


class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals.

    The train set is purged of observations that overlap with the test set.
    The test set is assumed to be contiguous (shuffle=False), with no training
    samples in between. An embargo period can be added to the training set to
    prevent leakage from information after the test set.
    """

    def __init__(self, n_splits=3, t1=None, pct_embargo=0.0):
        """
        Initialize the PurgedKFold splitter.

        Args:
            n_splits (int): The number of folds.
            t1 (pd.Series): A pandas Series where the index is the event start time
                and the value is the event end time.
            pct_embargo (float): The percentage of the dataset to be used as an
                embargo period, applied after the test set to prevent leakage.
        """
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pandas Series.")
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Args:
            X (pd.DataFrame): The data to split. Must have an index that matches t1.
            y: Ignored.
            groups: Ignored.

        Yields:
            tuple: A tuple of (train_indices, test_indices).
        """
        if not (X.index == self.t1.index).all():
            raise ValueError("X and t1 must have the same index.")

        indices = np.arange(X.shape[0])
        embargo_size = int(X.shape[0] * self.pct_embargo)

        test_ranges = [
            (i[0], i[-1] + 1)
            for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]

        for test_start_idx, test_end_idx in test_ranges:
            test_indices = indices[test_start_idx:test_end_idx]

            # --- Training set before test set ---
            test_start_time = self.t1.index[test_start_idx]
            train_indices_before_times = self.t1[self.t1 <= test_start_time].index
            train_indices_before = self.t1.index.searchsorted(
                train_indices_before_times
            )

            # --- Training set after test set ---
            latest_end_in_test = self.t1.iloc[test_indices].max()

            first_start_after_test_idx = self.t1.index.searchsorted(
                latest_end_in_test
            )

            train_indices_after = indices[
                first_start_after_test_idx + embargo_size :
            ]

            train_indices = np.concatenate((train_indices_before, train_indices_after))
            yield train_indices, test_indices
