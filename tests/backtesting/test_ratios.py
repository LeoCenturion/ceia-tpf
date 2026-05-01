import unittest

import numpy as np
import pandas as pd

from src.backtesting.ratios import (
    calmar_ratio,
    deflated_sharpe_ratio,
    path_to_returns,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Helpers that mirror how cpcv_runner.py builds paths
# ---------------------------------------------------------------------------


def _make_path(
    n_obs: int, drift: float = 0.001, vol: float = 0.02, signal: int = 1, seed: int = 0
) -> dict:
    """Synthetic CPCV path: prices from a log-normal process, constant signal."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(drift, vol, n_obs)
    prices = 100.0 * np.exp(np.cumsum(log_rets))
    signals = np.full(n_obs, signal, dtype=float)
    return {"y_true": prices, "y_pred": signals}


def _make_paths(
    n_paths: int, n_obs: int = 250, drift: float = 0.001, vol: float = 0.02
) -> list:
    return [_make_path(n_obs, drift=drift, vol=vol, seed=i) for i in range(n_paths)]


# ---------------------------------------------------------------------------
# path_to_returns
# ---------------------------------------------------------------------------


class TestPathToReturns(unittest.TestCase):
    def test_length_drops_first_nan(self):
        path = _make_path(100, signal=1)
        rets = path_to_returns(path)
        self.assertEqual(len(rets), 99)

    def test_zero_signal_gives_zero_returns(self):
        path = _make_path(50, signal=0)
        rets = path_to_returns(path)
        self.assertTrue((rets == 0.0).all())

    def test_negative_signal_flips_sign(self):
        path_pos = _make_path(100, signal=1, seed=7)
        path_neg = _make_path(100, signal=-1, seed=7)
        rets_pos = path_to_returns(path_pos)
        rets_neg = path_to_returns(path_neg)
        np.testing.assert_array_almost_equal(rets_pos.values, -rets_neg.values)

    def test_returns_index_has_no_nan(self):
        path = _make_path(200, seed=3)
        rets = path_to_returns(path)
        self.assertFalse(rets.isna().any())


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------


class TestProbabilisticSharpeRatio(unittest.TestCase):
    def test_output_is_probability(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            r = rng.normal(0.001, 0.02, 200)
            psr = probabilistic_sharpe_ratio(r)
            self.assertGreaterEqual(psr, 0.0)
            self.assertLessEqual(psr, 1.0)

    def test_positive_drift_gives_psr_above_half(self):
        rng = np.random.default_rng(1)
        # SR ≈ 0.5 per period; with 200 obs this should dominate
        r = rng.normal(0.01, 0.02, 200)
        self.assertGreater(probabilistic_sharpe_ratio(r, benchmark_sr=0.0), 0.5)

    def test_negative_drift_gives_psr_below_half(self):
        rng = np.random.default_rng(2)
        r = rng.normal(-0.01, 0.02, 200)
        self.assertLess(probabilistic_sharpe_ratio(r, benchmark_sr=0.0), 0.5)

    def test_higher_benchmark_lowers_psr(self):
        rng = np.random.default_rng(3)
        r = rng.normal(0.005, 0.02, 300)
        psr_low = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
        psr_high = probabilistic_sharpe_ratio(r, benchmark_sr=0.5)
        self.assertGreater(psr_low, psr_high)

    def test_too_few_observations_returns_nan(self):
        self.assertTrue(
            np.isnan(probabilistic_sharpe_ratio(np.array([0.01, -0.01, 0.02])))
        )

    def test_cpcv_path_roundtrip(self):
        """PSR accepts output of path_to_returns without error."""
        path = _make_path(300, drift=0.002, vol=0.02, signal=1, seed=10)
        rets = path_to_returns(path)
        psr = probabilistic_sharpe_ratio(rets, benchmark_sr=0.0)
        self.assertGreaterEqual(psr, 0.0)
        self.assertLessEqual(psr, 1.0)

    def test_consistent_with_normal_approximation(self):
        """For near-normal returns with large T, PSR ≈ Φ(SR_hat * sqrt(T))."""
        rng = np.random.default_rng(42)
        r = rng.normal(0.005, 0.1, 5000)  # large sample, nearly normal
        psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
        sr_hat = r.mean() / r.std(ddof=1)
        # SR_hat * sqrt(T-1) / sqrt(1 + ...) ≈ SR_hat * sqrt(T)
        naive_z = sr_hat * np.sqrt(len(r) - 1)
        from scipy.special import ndtr

        naive_psr = float(ndtr(naive_z))
        self.assertAlmostEqual(psr, naive_psr, places=1)

    # --- corner cases ---

    def test_constant_nonzero_returns_returns_nan(self):
        # std ≈ 0 due to floating-point noise; scipy skew/kurt return nan;
        # max(nan, 0.0) silently becomes 0.0 → old code returned 1.0 (wrong).
        r = np.array([0.01] * 50, dtype=float)
        self.assertTrue(np.isnan(probabilistic_sharpe_ratio(r)))

    def test_constant_zero_returns_returns_nan(self):
        r = np.zeros(50)
        self.assertTrue(np.isnan(probabilistic_sharpe_ratio(r)))

    def test_embedded_nans_are_filtered(self):
        # NaN values are removed before calculation; result must be finite.
        rng = np.random.default_rng(20)
        clean = rng.normal(0.005, 0.02, 100)
        dirty = clean.copy()
        dirty[::10] = np.nan  # inject 10 NaNs
        psr_clean = probabilistic_sharpe_ratio(clean)
        psr_dirty = probabilistic_sharpe_ratio(dirty)
        # Both should be valid probabilities; dirty has fewer obs so slightly different.
        self.assertTrue(np.isfinite(psr_dirty))
        self.assertGreaterEqual(psr_dirty, 0.0)
        self.assertLessEqual(psr_dirty, 1.0)
        self.assertTrue(np.isfinite(psr_clean))
        self.assertGreaterEqual(psr_clean, 0.0)
        self.assertLessEqual(psr_clean, 1.0)

    def test_exactly_four_observations(self):
        # n=4 is the minimum accepted; kurtosis denominator uses n-1=3 (no blow-up).
        r = np.array([0.01, -0.005, 0.02, 0.008])
        psr = probabilistic_sharpe_ratio(r)
        self.assertTrue(np.isfinite(psr))
        self.assertGreaterEqual(psr, 0.0)
        self.assertLessEqual(psr, 1.0)

    def test_negative_benchmark_increases_psr(self):
        # SR_hat > -0.2 is easier to beat than SR_hat > 0 → higher PSR.
        rng = np.random.default_rng(30)
        r = rng.normal(0.001, 0.02, 300)
        psr_zero = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
        psr_neg = probabilistic_sharpe_ratio(r, benchmark_sr=-0.2)
        self.assertGreater(psr_neg, psr_zero)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


class TestDeflatedSharpeRatio(unittest.TestCase):
    def test_output_is_probability(self):
        paths = _make_paths(5, n_obs=250)
        returns_list = [path_to_returns(p) for p in paths]
        dsr = deflated_sharpe_ratio(returns_list)
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)

    def test_empty_input_returns_nan(self):
        self.assertTrue(np.isnan(deflated_sharpe_ratio([])))

    def test_single_path_equals_psr_at_benchmark_zero(self):
        """With N=1, E[max_sr]=0 so DSR == PSR(benchmark_sr=0)."""
        rng = np.random.default_rng(5)
        r = rng.normal(0.003, 0.02, 300)
        psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
        dsr = deflated_sharpe_ratio([r], benchmark_sr=0.0)
        self.assertAlmostEqual(psr, dsr, places=10)

    def test_more_trials_inflates_benchmark(self):
        """Adding noise paths raises the benchmark and lowers DSR for the same best path."""
        rng = np.random.default_rng(99)
        # Strong signal: SR ≈ 0.5 per period — will always be the best path
        best = rng.normal(0.01, 0.02, 300)
        noise = [rng.normal(0.0, 0.02, 300) for _ in range(29)]

        # N=1: no multiple-testing correction → highest DSR
        dsr_1 = deflated_sharpe_ratio([best])
        # N=30: benchmark inflated by E[max of 30 std-normals]
        dsr_30 = deflated_sharpe_ratio([best] + noise)

        self.assertGreaterEqual(dsr_1, dsr_30)

    def test_strong_signal_survives_deflation(self):
        """A genuinely strong strategy should keep high DSR even among many paths."""
        rng = np.random.default_rng(11)
        # 19 noise paths + 1 strong path (SR ≈ 1.5 per period)
        noise = [rng.normal(0.0, 0.02, 400) for _ in range(19)]
        strong = rng.normal(0.03, 0.02, 400)
        dsr = deflated_sharpe_ratio(noise + [strong])
        self.assertGreater(dsr, 0.8)

    def test_cpcv_paths_end_to_end(self):
        """DSR works with the full CPCV-path → path_to_returns pipeline."""
        # 9 paths: C(9,2)=36 splits, comparable to a real CPCV run
        paths = _make_paths(9, n_obs=400, drift=0.002, vol=0.02)
        returns_list = [path_to_returns(p) for p in paths]
        dsr = deflated_sharpe_ratio(returns_list)
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)

    def test_dsr_leq_psr_same_benchmark(self):
        """DSR ≤ PSR of the best path at the same base benchmark (N≥2)."""
        paths = _make_paths(8, n_obs=300, drift=0.003, vol=0.02)
        returns_list = [path_to_returns(p) for p in paths]
        best_idx = int(np.argmax([r.mean() / r.std() for r in returns_list]))
        psr_best = probabilistic_sharpe_ratio(returns_list[best_idx], benchmark_sr=0.0)
        dsr = deflated_sharpe_ratio(returns_list, benchmark_sr=0.0)
        self.assertLessEqual(dsr, psr_best + 1e-12)

    def test_all_short_series_returns_nan(self):
        """If every path has fewer than 4 observations, DSR is NaN."""
        tiny_paths = [np.array([0.01, -0.01, 0.02])] * 5
        self.assertTrue(np.isnan(deflated_sharpe_ratio(tiny_paths)))

    # --- corner cases ---

    def test_constant_returns_in_one_path_are_skipped(self):
        # A constant path is degenerate; DSR should still work using the other paths.
        rng = np.random.default_rng(40)
        normal = [rng.normal(0.002, 0.02, 300) for _ in range(4)]
        constant = np.full(300, 0.01)
        dsr = deflated_sharpe_ratio(normal + [constant])
        self.assertTrue(np.isfinite(dsr))
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)

    def test_all_constant_returns_returns_nan(self):
        constant_paths = [np.full(100, 0.01)] * 5
        self.assertTrue(np.isnan(deflated_sharpe_ratio(constant_paths)))

    def test_unequal_path_lengths(self):
        # Each path is processed independently so different lengths are fine.
        rng = np.random.default_rng(50)
        short = rng.normal(0.002, 0.02, 100)
        medium = rng.normal(0.002, 0.02, 300)
        long_ = rng.normal(0.002, 0.02, 600)
        dsr = deflated_sharpe_ratio([short, medium, long_])
        self.assertTrue(np.isfinite(dsr))
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio(unittest.TestCase):
    def test_positive_drift_gives_positive_sharpe(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.01, 0.02, 500)
        self.assertGreater(sharpe_ratio(r), 0.0)

    def test_negative_drift_gives_negative_sharpe(self):
        rng = np.random.default_rng(1)
        r = rng.normal(-0.01, 0.02, 500)
        self.assertLess(sharpe_ratio(r), 0.0)

    def test_zero_std_returns_nan(self):
        r = np.full(50, 0.005)
        self.assertTrue(np.isnan(sharpe_ratio(r)))

    def test_fewer_than_two_obs_returns_nan(self):
        self.assertTrue(np.isnan(sharpe_ratio(np.array([0.01]))))
        self.assertTrue(np.isnan(sharpe_ratio(np.array([]))))

    def test_periods_per_year_scales_annualized_value(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.005, 0.02, 300)
        sr_daily = sharpe_ratio(r, periods_per_year=365)
        sr_hourly = sharpe_ratio(r, periods_per_year=365 * 24)
        self.assertGreater(sr_hourly, sr_daily)

    def test_nan_values_are_filtered(self):
        rng = np.random.default_rng(3)
        clean = rng.normal(0.005, 0.02, 200)
        dirty = clean.copy()
        dirty[::20] = np.nan
        sr_clean = sharpe_ratio(clean)
        sr_dirty = sharpe_ratio(dirty)
        # Both finite and same sign
        self.assertTrue(np.isfinite(sr_dirty))
        self.assertEqual(np.sign(sr_clean), np.sign(sr_dirty))

    def test_accepts_pandas_series(self):
        rng = np.random.default_rng(4)
        r = pd.Series(rng.normal(0.005, 0.02, 100))
        result = sharpe_ratio(r)
        self.assertTrue(np.isfinite(result))


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio(unittest.TestCase):
    def test_no_drawdown_returns_nan(self):
        # Monotonically increasing prices → no drawdown
        r = np.full(50, 0.01)  # constant positive returns
        self.assertTrue(np.isnan(calmar_ratio(r)))

    def test_fewer_than_two_obs_returns_nan(self):
        self.assertTrue(np.isnan(calmar_ratio(np.array([0.01]))))
        self.assertTrue(np.isnan(calmar_ratio(np.array([]))))

    def test_with_drawdown_returns_finite(self):
        rng = np.random.default_rng(5)
        r = rng.normal(0.001, 0.02, 300)
        cr = calmar_ratio(r)
        self.assertTrue(np.isfinite(cr))

    def test_positive_mean_positive_calmar_when_drawdown_exists(self):
        rng = np.random.default_rng(6)
        r = rng.normal(0.01, 0.03, 500)
        # Inject a large drawdown so max_dd > 0
        r[100:110] = -0.1
        cr = calmar_ratio(r)
        if np.isfinite(cr):
            # sign should match the sign of annualized return
            ann_ret = r.mean() * (365 * 24)
            self.assertEqual(np.sign(cr), np.sign(ann_ret))

    def test_accepts_pandas_series(self):
        rng = np.random.default_rng(7)
        r = pd.Series(rng.normal(0.001, 0.02, 300))
        r.iloc[50] = -0.15  # force a drawdown
        cr = calmar_ratio(r)
        self.assertTrue(np.isfinite(cr))

    def test_nan_values_are_filtered(self):
        rng = np.random.default_rng(8)
        r = rng.normal(0.001, 0.02, 300)
        r[50] = -0.2  # ensure drawdown
        dirty = r.copy()
        dirty[::30] = np.nan
        cr_dirty = calmar_ratio(dirty)
        self.assertTrue(np.isfinite(cr_dirty))

    def test_periods_per_year_scales_result(self):
        rng = np.random.default_rng(9)
        r = rng.normal(0.001, 0.02, 300)
        r[50] = -0.2  # ensure drawdown
        cr_daily = calmar_ratio(r, periods_per_year=365)
        cr_hourly = calmar_ratio(r, periods_per_year=365 * 24)
        # Calmar scales linearly with periods_per_year (same max_dd, scaled return)
        if np.isfinite(cr_daily) and np.isfinite(cr_hourly):
            self.assertAlmostEqual(cr_hourly / cr_daily, 24.0, places=5)


if __name__ == "__main__":
    unittest.main()
