import unittest
from math import comb

import numpy as np
import pandas as pd

from src.backtesting.cscv import build_performance_matrix, compute_pbo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_M(T: int = 200, N: int = 10, seed: int = 0) -> pd.DataFrame:
    """Random performance matrix with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=T, freq="1h")
    data = rng.normal(0.0, 0.01, size=(T, N))
    return pd.DataFrame(data, index=dates, columns=[f"t{i}" for i in range(N)])


def _make_oos_df_labels(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """OOS DataFrame with binary classification labels."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="1h")
    y_true = rng.integers(0, 2, n).astype(float)
    y_pred = rng.integers(0, 2, n).astype(float)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=dates)


def _make_oos_df_prices(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """OOS DataFrame with close prices as y_true and binary signals as y_pred."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="1h")
    prices = 10000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    signals = rng.integers(0, 2, n).astype(float)
    return pd.DataFrame({"y_true": prices, "y_pred": signals}, index=dates)


# ---------------------------------------------------------------------------
# TestComputePBO
# ---------------------------------------------------------------------------

class TestComputePBO(unittest.TestCase):

    def test_pbo_is_in_unit_interval(self):
        M = _make_M(T=160, N=8)
        result = compute_pbo(M, S=8)
        self.assertGreaterEqual(result["pbo"], 0.0)
        self.assertLessEqual(result["pbo"], 1.0)

    def test_pbo_near_zero_for_dominant_strategy(self):
        """One strategy returns +0.1 per step; all others return ~0."""
        rng = np.random.default_rng(42)
        T, N = 160, 10
        dates = pd.date_range("2022-01-01", periods=T, freq="1h")
        data = rng.normal(0.0, 0.001, size=(T, N))
        data[:, 0] = 0.01  # dominant strategy
        M = pd.DataFrame(data, index=dates)
        result = compute_pbo(M, S=8)
        # Dominant strategy should win IS and OOS → most lambdas > 0 → PBO near 0
        self.assertLess(result["pbo"], 0.3)

    def test_pbo_near_half_for_uniform_strategies(self):
        """All strategies i.i.d. → each wins IS ~1/N of the time → PBO ≈ 0.5."""
        rng = np.random.default_rng(0)
        T, N, S = 320, 20, 8
        dates = pd.date_range("2022-01-01", periods=T, freq="1h")
        data = rng.normal(0.0, 0.01, size=(T, N))
        M = pd.DataFrame(data, index=dates)
        result = compute_pbo(M, S=S)
        # With i.i.d. strategies, IS winner is random → ~50% chance OOS rank < 0.5
        self.assertAlmostEqual(result["pbo"], 0.5, delta=0.25)

    def test_requires_even_S(self):
        M = _make_M(T=100, N=5)
        with self.assertRaises(ValueError):
            compute_pbo(M, S=5)

    def test_S_must_be_at_least_2(self):
        M = _make_M(T=100, N=5)
        with self.assertRaises(ValueError):
            compute_pbo(M, S=1)

    def test_n_combinations_matches_binomial(self):
        S = 6
        M = _make_M(T=120, N=4)
        result = compute_pbo(M, S=S)
        expected = comb(S, S // 2)
        self.assertEqual(result["n_combinations"], expected)

    def test_lambda_values_length_matches_n_combinations(self):
        S = 4
        M = _make_M(T=80, N=5)
        result = compute_pbo(M, S=S)
        self.assertEqual(len(result["lambda_values"]), result["n_combinations"])

    def test_lambda_values_are_finite(self):
        M = _make_M(T=80, N=5)
        result = compute_pbo(M, S=4)
        for lv in result["lambda_values"]:
            self.assertTrue(np.isfinite(lv), f"Non-finite lambda: {lv}")

    def test_custom_metric_fn(self):
        M = _make_M(T=80, N=4)
        mean_fn = lambda arr: float(np.mean(arr))
        result = compute_pbo(M, S=4, metric_fn=mean_fn)
        self.assertGreaterEqual(result["pbo"], 0.0)
        self.assertLessEqual(result["pbo"], 1.0)

    def test_T_less_than_S_raises(self):
        M = _make_M(T=4, N=3)
        with self.assertRaises(ValueError):
            compute_pbo(M, S=8)


# ---------------------------------------------------------------------------
# TestBuildPerformanceMatrix
# ---------------------------------------------------------------------------

class TestBuildPerformanceMatrix(unittest.TestCase):

    def test_single_trial_returns_single_column(self):
        oos = {"trial_0": _make_oos_df_labels(200)}
        M = build_performance_matrix(oos, mode="hit", freq="1h")
        self.assertEqual(M.shape[1], 1)
        self.assertIn("trial_0", M.columns)

    def test_hit_mode_values_in_zero_one(self):
        oos = {f"t{i}": _make_oos_df_labels(200, seed=i) for i in range(3)}
        M = build_performance_matrix(oos, mode="hit", freq="1h")
        self.assertTrue(((M >= 0.0) & (M <= 1.0)).all().all())

    def test_signal_return_mode_produces_finite_values(self):
        oos = {"t0": _make_oos_df_prices(300), "t1": _make_oos_df_prices(300, seed=1)}
        M = build_performance_matrix(oos, mode="signal_return", freq="1h")
        self.assertFalse(M.empty)
        self.assertTrue(np.isfinite(M.values).all())

    def test_inner_join_aligns_different_length_series(self):
        dates_long = pd.date_range("2022-01-01", periods=500, freq="1h")
        dates_short = pd.date_range("2022-06-01", periods=200, freq="1h")
        rng = np.random.default_rng(0)
        df_long = pd.DataFrame(
            {"y_true": rng.integers(0, 2, 500).astype(float),
             "y_pred": rng.integers(0, 2, 500).astype(float)},
            index=dates_long,
        )
        df_short = pd.DataFrame(
            {"y_true": rng.integers(0, 2, 200).astype(float),
             "y_pred": rng.integers(0, 2, 200).astype(float)},
            index=dates_short,
        )
        M = build_performance_matrix({"long": df_long, "short": df_short}, mode="hit", freq="1h")
        # Common overlap is dates_short ∩ dates_long (200 rows at most)
        self.assertLessEqual(len(M), 200)
        self.assertEqual(M.shape[1], 2)

    def test_auto_mode_detects_labels(self):
        oos = {"t0": _make_oos_df_labels(200)}
        M = build_performance_matrix(oos, mode="auto", freq="1h")
        # Labels → hit mode → values in [0, 1]
        self.assertTrue(((M >= 0.0) & (M <= 1.0)).all().all())

    def test_auto_mode_detects_prices(self):
        oos = {"t0": _make_oos_df_prices(300)}
        M = build_performance_matrix(oos, mode="auto", freq="1h")
        # Prices → signal_return mode → values can be negative
        self.assertFalse(M.empty)

    def test_empty_input_returns_empty_dataframe(self):
        M = build_performance_matrix({}, mode="hit", freq="1h")
        self.assertTrue(M.empty)

    def test_column_names_match_trial_ids(self):
        trial_ids = ["trial_a", "trial_b", "trial_c"]
        oos = {tid: _make_oos_df_labels(200, seed=i) for i, tid in enumerate(trial_ids)}
        M = build_performance_matrix(oos, mode="hit", freq="1h")
        self.assertEqual(sorted(M.columns.tolist()), sorted(trial_ids))

    def test_no_nans_after_build(self):
        oos = {f"t{i}": _make_oos_df_labels(200, seed=i) for i in range(4)}
        M = build_performance_matrix(oos, mode="hit", freq="1h")
        self.assertFalse(M.isnull().any().any())


if __name__ == "__main__":
    unittest.main()
