"""
Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).

References:
  Bailey & López de Prado (2012) - "The Sharpe Ratio Efficient Frontier"
  Bailey & López de Prado (2014) - "The Deflated Sharpe Ratio"
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr

_EULER_GAMMA = 0.5772156649015328


def path_to_returns(path: Dict[str, np.ndarray]) -> pd.Series:
    """Convert a CPCV path dict (y_true prices, y_pred signals) to a return series."""
    prices = pd.Series(path["y_true"], dtype=float)
    signals = pd.Series(path["y_pred"], dtype=float)
    returns = prices.pct_change() * signals
    return returns.dropna()


def _sr_std(sr_hat: float, skew: float, kurtosis: float, n_obs: int) -> float:
    """
    Asymptotic standard deviation of the SR estimator (Mertens 2002).

    kurtosis is Pearson (non-excess) kurtosis: 3 for Gaussian returns.
    """
    variance = (1.0 - skew * sr_hat + (kurtosis - 1.0) / 4.0 * sr_hat**2) / (n_obs - 1)
    return float(np.sqrt(max(variance, 0.0)))


def probabilistic_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    benchmark_sr: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR).

    Estimates P(SR_true > benchmark_sr) using the asymptotic distribution of
    the SR estimator, correcting for non-Gaussian returns via skewness and kurtosis.

    Parameters
    ----------
    returns:
        Per-period return series (e.g. from ``path_to_returns``).
    benchmark_sr:
        Benchmark Sharpe ratio in per-period units (default 0).

    Returns
    -------
    float in [0, 1]
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4:
        return np.nan

    # Constant returns → std ≈ 0, scipy skew/kurtosis return nan with RuntimeWarning;
    # max(nan, 0.0) in _sr_std silently produces sigma=0 and a wrong result.
    if np.ptp(r) == 0.0:
        return np.nan

    sr_hat = r.mean() / r.std(ddof=1)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # Pearson kurtosis (3 for normal)
    sigma = _sr_std(sr_hat, skew, kurt, n)

    if sigma == 0.0:
        return 1.0 if sr_hat > benchmark_sr else 0.0

    return float(ndtr((sr_hat - benchmark_sr) / sigma))


def _expected_max_sr(n_trials: int) -> float:
    """
    Expected maximum of n_trials i.i.d. standard-normal variables.

    Approximation from Bailey & López de Prado (2014):
        E[max] ≈ (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))

    Returns 0 for N ≤ 1 (no multiple-testing inflation).
    """
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    returns_list: List[Union[np.ndarray, pd.Series]],
    benchmark_sr: float = 0.0,
) -> float:
    """
    Deflated Sharpe Ratio (DSR).

    PSR for the best path among N CPCV paths, using a benchmark inflated by
    the expected maximum SR under the null hypothesis. This corrects for the
    multiple-testing bias introduced when selecting the best path from a CPCV run.

    Parameters
    ----------
    returns_list:
        Per-period return series for each CPCV path (use ``path_to_returns``).
    benchmark_sr:
        Base benchmark in per-period units (default 0).

    Returns
    -------
    float in [0, 1]: probability that the best path's SR is genuine.
    """
    if not returns_list:
        return np.nan

    sr_hats: List[float] = []
    sigma_sr_list: List[float] = []

    for ret in returns_list:
        r = np.asarray(ret, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 4 or np.ptp(r) == 0.0:
            continue
        sr = r.mean() / r.std(ddof=1)
        skew = float(stats.skew(r))
        kurt = float(stats.kurtosis(r, fisher=False))
        sr_hats.append(sr)
        sigma_sr_list.append(_sr_std(sr, skew, kurt, len(r)))

    if not sr_hats:
        return np.nan

    best_idx = int(np.argmax(sr_hats))
    best_returns = np.asarray(returns_list[best_idx], dtype=float)
    best_returns = best_returns[np.isfinite(best_returns)]

    sigma_bar = float(np.mean(sigma_sr_list))
    sr_star = benchmark_sr + sigma_bar * _expected_max_sr(len(sr_hats))

    return probabilistic_sharpe_ratio(best_returns, benchmark_sr=sr_star)
