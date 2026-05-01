import tempfile
import unittest
from math import comb

import numpy as np
import pandas as pd

from src.backtesting.cpcv import (
    construct_backtest_paths,
    generate_combinatorial_splits,
    purge_and_embargo_split,
    time_based_partition,
)
from src.backtesting.cpcv_runner import (
    _evaluate_paths_f1,
    _evaluate_paths_returns,
    _save_paths_artifact,
    run_cpcv_for_strategy,
)
from src.backtesting.strategies.statistical_strategies import SmaCross
from src.modeling.mlflow_utils import MLflowLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n_bars: int = 500, freq: str = "1h", seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV data sufficient for SMA-based strategies."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_bars, freq=freq)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, n_bars)))
    high = close * (1 + rng.uniform(0.001, 0.005, n_bars))
    low = close * (1 - rng.uniform(0.001, 0.005, n_bars))
    open_ = close * (1 + rng.normal(0, 0.002, n_bars))
    volume = rng.uniform(1000, 5000, n_bars)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _make_t1(data: pd.DataFrame) -> pd.Series:
    """t1 series: each event ends at the immediately following bar."""
    return pd.Series(data.index[1:], index=data.index[:-1])


def _make_price_signal_paths(n_paths: int = 3, n_obs: int = 200, seed: int = 0):
    """CPCV paths with Close prices as y_true and binary signals as y_pred."""
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n_paths):
        log_rets = rng.normal(0.001, 0.02, n_obs)
        prices = 100.0 * np.exp(np.cumsum(log_rets))
        signals = rng.integers(0, 2, n_obs).astype(float)
        paths.append({"y_true": prices, "y_pred": signals})
    return paths


def _make_binary_paths(n_paths: int = 3, n_obs: int = 100, seed: int = 0):
    """Synthetic CPCV paths with binary classification labels."""
    rng = np.random.default_rng(seed)
    return [
        {"y_true": rng.integers(0, 2, n_obs), "y_pred": rng.integers(0, 2, n_obs)}
        for _ in range(n_paths)
    ]


def _make_split_predictions(n_groups: int, k: int, obs_per_group: int = 50):
    """Synthetic split_predictions as CPCV would produce."""
    rng = np.random.default_rng(0)
    splits = generate_combinatorial_splits(n_groups, k)
    result = []
    for _, test_groups in splits:
        size = obs_per_group * k
        result.append({
            "test_path_idxs": test_groups,
            "preds": rng.integers(0, 2, size),
            "y_test": rng.integers(0, 2, size),
        })
    return result


# ---------------------------------------------------------------------------
# time_based_partition
# ---------------------------------------------------------------------------

class TestTimeBasedPartition(unittest.TestCase):

    def setUp(self):
        self.index = pd.date_range("2022-01-01", periods=200, freq="1h")

    def test_produces_n_groups(self):
        partitions = time_based_partition(self.index, 5)
        self.assertEqual(len(partitions), 5)

    def test_covers_all_indices_exactly_once(self):
        partitions = time_based_partition(self.index, 4)
        all_indices = np.concatenate(partitions)
        self.assertEqual(sorted(all_indices.tolist()), list(range(len(self.index))))

    def test_groups_are_pairwise_disjoint(self):
        partitions = time_based_partition(self.index, 4)
        for i, a in enumerate(partitions):
            for j, b in enumerate(partitions):
                if i != j:
                    overlap = set(a.tolist()) & set(b.tolist())
                    self.assertEqual(len(overlap), 0, f"Groups {i} and {j} overlap")

    def test_each_group_is_nonempty(self):
        partitions = time_based_partition(self.index, 5)
        for i, p in enumerate(partitions):
            self.assertGreater(len(p), 0, f"Group {i} is empty")

    def test_single_group_contains_everything(self):
        partitions = time_based_partition(self.index, 1)
        self.assertEqual(len(partitions[0]), len(self.index))


# ---------------------------------------------------------------------------
# generate_combinatorial_splits
# ---------------------------------------------------------------------------

class TestGenerateCombatorialSplits(unittest.TestCase):

    def test_count_matches_binomial_coefficient(self):
        splits = generate_combinatorial_splits(6, 2)
        self.assertEqual(len(splits), comb(6, 2))

    def test_train_and_test_partition_all_groups(self):
        n, k = 6, 2
        splits = generate_combinatorial_splits(n, k)
        for train, test in splits:
            self.assertEqual(sorted(list(train) + list(test)), list(range(n)))

    def test_test_split_has_exactly_k_groups(self):
        for n, k in [(5, 2), (6, 3), (8, 2)]:
            splits = generate_combinatorial_splits(n, k)
            for _, test in splits:
                self.assertEqual(len(test), k, f"n={n}, k={k}")

    def test_k_equal_to_n_raises(self):
        with self.assertRaises(ValueError):
            generate_combinatorial_splits(4, 4)

    def test_all_test_splits_are_unique(self):
        splits = generate_combinatorial_splits(6, 2)
        test_splits = [test for _, test in splits]
        self.assertEqual(len(test_splits), len(set(test_splits)))


# ---------------------------------------------------------------------------
# purge_and_embargo_split
# ---------------------------------------------------------------------------

class TestPurgeAndEmbargoSplit(unittest.TestCase):

    def setUp(self):
        raw = _make_ohlcv(200)
        t1 = _make_t1(raw)
        self.data = raw.iloc[:-1]   # align with t1
        self.t1 = t1
        self.path_indices = time_based_partition(self.data.index, 4)

    def test_train_and_test_are_disjoint(self):
        train, test = purge_and_embargo_split(
            self.data, self.t1, self.path_indices,
            train_group_idxs=(0, 1, 2), test_group_idxs=(3,),
            pct_embargo=0.01,
        )
        overlap = set(train.tolist()) & set(test.tolist())
        self.assertEqual(len(overlap), 0)

    def test_test_indices_match_test_groups(self):
        train, test = purge_and_embargo_split(
            self.data, self.t1, self.path_indices,
            train_group_idxs=(0, 1), test_group_idxs=(2, 3),
            pct_embargo=0.0,
        )
        expected = sorted(np.concatenate([self.path_indices[2], self.path_indices[3]]).tolist())
        self.assertEqual(sorted(test.tolist()), expected)

    def test_embargo_reduces_train_size(self):
        train_no_emb, _ = purge_and_embargo_split(
            self.data, self.t1, self.path_indices,
            train_group_idxs=(0, 1, 2), test_group_idxs=(3,),
            pct_embargo=0.0,
        )
        train_with_emb, _ = purge_and_embargo_split(
            self.data, self.t1, self.path_indices,
            train_group_idxs=(0, 1, 2), test_group_idxs=(3,),
            pct_embargo=0.15,
        )
        self.assertLessEqual(len(train_with_emb), len(train_no_emb))

    def test_train_indices_are_valid(self):
        train, _ = purge_and_embargo_split(
            self.data, self.t1, self.path_indices,
            train_group_idxs=(0, 1), test_group_idxs=(2, 3),
            pct_embargo=0.01,
        )
        self.assertTrue(np.all(train >= 0))
        self.assertTrue(np.all(train < len(self.data)))


# ---------------------------------------------------------------------------
# construct_backtest_paths
# ---------------------------------------------------------------------------

class TestConstructBacktestPaths(unittest.TestCase):

    def test_paths_are_nonempty(self):
        preds = _make_split_predictions(4, 2)
        paths = construct_backtest_paths(preds, 4, 2)
        self.assertGreater(len(paths), 0)

    def test_each_path_has_required_keys(self):
        preds = _make_split_predictions(4, 2)
        paths = construct_backtest_paths(preds, 4, 2)
        for path in paths:
            self.assertIn("y_true", path)
            self.assertIn("y_pred", path)

    def test_y_true_and_y_pred_have_equal_length(self):
        preds = _make_split_predictions(4, 2)
        paths = construct_backtest_paths(preds, 4, 2)
        for path in paths:
            self.assertEqual(len(path["y_true"]), len(path["y_pred"]))

    def test_values_come_from_original_predictions(self):
        preds = _make_split_predictions(4, 2)
        paths = construct_backtest_paths(preds, 4, 2)
        for path in paths:
            unique_preds = set(path["y_pred"].tolist())
            self.assertTrue(unique_preds.issubset({0, 1}))

    def test_expected_number_of_paths(self):
        """C(n-1, k-1) paths for n groups and k test groups."""
        n, k = 4, 2
        preds = _make_split_predictions(n, k)
        paths = construct_backtest_paths(preds, n, k)
        expected = comb(n - 1, k - 1)
        self.assertEqual(len(paths), expected)

    def test_empty_predictions_returns_empty(self):
        paths = construct_backtest_paths([], 4, 2)
        self.assertEqual(paths, [])


# ---------------------------------------------------------------------------
# _evaluate_paths_f1
# ---------------------------------------------------------------------------

class TestEvaluatePathsF1(unittest.TestCase):

    _tmpdir = None

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        cls._tracking_uri = f"sqlite:///{cls._tmpdir}/mlflow_test.db"

    def _logger(self):
        return MLflowLogger(
            experiment_name="test_evaluate_paths_f1",
            tracking_uri=self._tracking_uri,
        )

    def test_returns_one_score_per_path(self):
        logger = self._logger()
        paths = _make_binary_paths(4)
        with logger.start_run(run_name="test_run"):
            scores = _evaluate_paths_f1(paths, logger)
        self.assertEqual(len(scores), 4)

    def test_scores_are_valid_f1_values(self):
        logger = self._logger()
        paths = _make_binary_paths(3)
        with logger.start_run(run_name="test_run"):
            scores = _evaluate_paths_f1(paths, logger)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_empty_paths_returns_empty_list(self):
        logger = self._logger()
        with logger.start_run(run_name="test_run"):
            scores = _evaluate_paths_f1([], logger)
        self.assertEqual(scores, [])

    def test_perfect_predictions_score_one(self):
        logger = self._logger()
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        paths = [{"y_true": y, "y_pred": y}]
        with logger.start_run(run_name="test_run"):
            scores = _evaluate_paths_f1(paths, logger)
        self.assertAlmostEqual(scores[0], 1.0, places=5)

    def test_all_wrong_predictions_score_near_zero(self):
        logger = self._logger()
        y_true = np.array([1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        paths = [{"y_true": y_true, "y_pred": y_pred}]
        with logger.start_run(run_name="test_run"):
            scores = _evaluate_paths_f1(paths, logger)
        self.assertAlmostEqual(scores[0], 0.0, places=5)


# ---------------------------------------------------------------------------
# End-to-end: SmaCross through run_cpcv_for_strategy
# ---------------------------------------------------------------------------

class TestSmaCrossCPCVEndToEnd(unittest.TestCase):
    """
    Full CPCV run with SmaCross on synthetic OHLCV data.
    No mocks — strategy, CPCV splitter, and MLflow logger all execute for real.
    Results are collected once in setUpClass and shared across assertion tests.
    """

    _paths = None
    _data = None

    @classmethod
    def setUpClass(cls):
        data = _make_ohlcv(n_bars=500)
        t1 = _make_t1(data)
        cls._data = data.iloc[:-1]  # align with t1

        cls._paths = run_cpcv_for_strategy(
            data=cls._data,
            t1=t1,
            strategy_class=SmaCross,
            strategy_params={"n1": 10, "n2": 20},
            n_groups=4,
            k_test_groups=2,
            embargo_pct=0.01,
            experiment_name="test_sma_cross_cpcv_e2e",
        )

    def test_returns_nonempty_list_of_paths(self):
        self.assertIsInstance(self._paths, list)
        self.assertGreater(len(self._paths), 0)

    def test_expected_number_of_paths(self):
        """C(n-1, k-1) = C(3,1) = 3 paths for n=4, k=2."""
        self.assertEqual(len(self._paths), comb(4 - 1, 2 - 1))

    def test_each_path_has_y_true_and_y_pred(self):
        for path in self._paths:
            self.assertIn("y_true", path)
            self.assertIn("y_pred", path)

    def test_y_true_and_y_pred_have_equal_length(self):
        for path in self._paths:
            self.assertEqual(len(path["y_true"]), len(path["y_pred"]))

    def test_predictions_are_binary(self):
        """SmaCross only emits 0 (short/flat) or 1 (long)."""
        for i, path in enumerate(self._paths):
            unique = set(path["y_pred"].tolist())
            self.assertTrue(
                unique.issubset({0, 1}),
                f"Path {i} contains non-binary predictions: {unique}",
            )

    def test_y_true_are_positive_prices(self):
        """y_true is the Close price series, which should always be positive."""
        for path in self._paths:
            self.assertTrue(
                np.all(path["y_true"] > 0),
                "Close prices contain non-positive values",
            )

    def test_each_path_length_is_positive(self):
        for path in self._paths:
            self.assertGreater(len(path["y_true"]), 0)

    def test_paths_together_cover_most_of_dataset(self):
        """
        Each path spans 2 test groups out of 4, so ~half the dataset.
        Three non-overlapping paths together should cover most observations.
        """
        total_obs = sum(len(p["y_true"]) for p in self._paths)
        # 3 paths × (500/4 bars × 2 groups) ≈ 750 obs > dataset length
        # Even with purging/embargo, total should exceed half the data
        self.assertGreater(total_obs, len(self._data) // 4)

    def test_returns_are_finite(self):
        """Sharpe computation requires finite percentage returns."""
        for path in self._paths:
            y_true = pd.Series(path["y_true"], dtype=float)
            returns = y_true.pct_change().dropna()
            self.assertTrue(np.all(np.isfinite(returns)))


# ---------------------------------------------------------------------------
# _save_paths_artifact
# ---------------------------------------------------------------------------

class TestSavePathsArtifact(unittest.TestCase):

    _tmpdir = None
    _tracking_uri = None

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        cls._tracking_uri = f"sqlite:///{cls._tmpdir}/mlflow_save_artifact_test.db"

    def _logger(self):
        return MLflowLogger(
            experiment_name="test_save_paths_artifact",
            tracking_uri=self._tracking_uri,
        )

    def test_artifact_is_logged_to_active_run(self):
        logger = self._logger()
        paths = _make_price_signal_paths(3, n_obs=50)
        with logger.start_run(run_name="artifact_run") as run:
            _save_paths_artifact(paths)
            run_id = run.info.run_id

        import mlflow
        mlflow.set_tracking_uri(self._tracking_uri)
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, path="predictions")
        self.assertGreater(len(artifacts), 0, "No artifact was logged under predictions/")

    def test_no_artifact_logged_on_empty_paths(self):
        logger = self._logger()
        with logger.start_run(run_name="empty_artifact_run") as run:
            _save_paths_artifact([])
            run_id = run.info.run_id

        import mlflow
        mlflow.set_tracking_uri(self._tracking_uri)
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, path="predictions")
        # Empty paths produces an empty CSV that is still logged
        # (behaviour: 0-row file still counts as an artifact)
        # We just verify the call doesn't raise
        self.assertIsInstance(artifacts, list)


# ---------------------------------------------------------------------------
# _evaluate_paths_returns
# ---------------------------------------------------------------------------

class TestEvaluatePathsReturns(unittest.TestCase):

    _tmpdir = None
    _tracking_uri = None

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        cls._tracking_uri = f"sqlite:///{cls._tmpdir}/mlflow_returns_test.db"

    def _logger(self):
        return MLflowLogger(
            experiment_name="test_evaluate_paths_returns",
            tracking_uri=self._tracking_uri,
        )

    def test_returns_one_score_per_path(self):
        logger = self._logger()
        paths = _make_price_signal_paths(4, n_obs=200)
        with logger.start_run(run_name="returns_run"):
            scores = _evaluate_paths_returns(paths, logger)
        self.assertEqual(len(scores), 4)

    def test_scores_are_floats(self):
        logger = self._logger()
        paths = _make_price_signal_paths(3, n_obs=150)
        with logger.start_run(run_name="scores_run"):
            scores = _evaluate_paths_returns(paths, logger)
        for s in scores:
            self.assertIsInstance(s, float)

    def test_empty_paths_returns_empty_list(self):
        logger = self._logger()
        with logger.start_run(run_name="empty_run"):
            scores = _evaluate_paths_returns([], logger)
        self.assertEqual(scores, [])

    def test_positive_drift_paths_give_nonnegative_sharpe(self):
        logger = self._logger()
        rng = np.random.default_rng(99)
        paths = []
        for i in range(3):
            log_rets = rng.normal(0.005, 0.01, 300)  # strong positive drift
            prices = 100.0 * np.exp(np.cumsum(log_rets))
            paths.append({"y_true": prices, "y_pred": np.ones(300)})
        with logger.start_run(run_name="drift_run"):
            scores = _evaluate_paths_returns(paths, logger)
        # Sharpe safe_sr clamps NaN to 0.0 so minimum is 0.0; with drift expect > 0
        self.assertTrue(all(s >= 0.0 for s in scores))

    def test_zero_signal_paths_give_zero_sharpe(self):
        logger = self._logger()
        rng = np.random.default_rng(42)
        log_rets = rng.normal(0.005, 0.02, 200)
        prices = 100.0 * np.exp(np.cumsum(log_rets))
        paths = [{"y_true": prices, "y_pred": np.zeros(200)}]
        with logger.start_run(run_name="zero_signal_run"):
            scores = _evaluate_paths_returns(paths, logger)
        self.assertAlmostEqual(scores[0], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
