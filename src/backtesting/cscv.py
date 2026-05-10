"""
Combinatorially Symmetric Cross-Validation (CSCV) and Probability of Backtest
Overfitting (PBO) — Bailey et al. (2014) / López de Prado framework.

Reference: docs/cspv.md
"""

import logging
from math import comb
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.backtesting.cpcv import generate_combinatorial_splits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sharpe_ratio(series: np.ndarray, periods_per_year: int = 365 * 24) -> float:
    r = np.asarray(series, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    std = r.std(ddof=1)
    return 0.0 if std == 0.0 else float(r.mean() / std * np.sqrt(periods_per_year))


def _divide_into_blocks(T: int, S: int) -> list[np.ndarray]:
    """Divide T indices into S equal-ish non-overlapping blocks."""
    indices = np.arange(T)
    block_size = T // S
    blocks = []
    start = 0
    for i in range(S):
        end = start + block_size + (1 if i < T % S else 0)
        blocks.append(indices[start:end])
        start = end
    return blocks


# ---------------------------------------------------------------------------
# Main CSCV / PBO
# ---------------------------------------------------------------------------

def compute_pbo(
    M: pd.DataFrame,
    S: int = 16,
    metric_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> dict:
    """
    Compute Probability of Backtest Overfitting (PBO) from a performance matrix.

    Parameters
    ----------
    M : pd.DataFrame, shape (T, N)
        Performance series: rows = time observations, columns = strategies/trials.
        Values should be per-timestep performance (e.g., returns, hit indicators).
    S : int
        Number of sub-matrices (blocks). Must be even and >= 2.
        C(S, S/2) combinations are evaluated; S=16 gives 12,870 combinations.
    metric_fn : callable, optional
        Function that maps a 1-D array of performance values to a scalar metric.
        Defaults to annualised Sharpe ratio.

    Returns
    -------
    dict with keys:
        "pbo"           : float in [0, 1] — estimated probability of overfitting
        "lambda_values" : list[float] — one logit per combination
        "n_combinations": int — number of IS/OOS splits evaluated
    """
    if S % 2 != 0:
        raise ValueError(f"S must be even, got {S}")
    if S < 2:
        raise ValueError(f"S must be >= 2, got {S}")

    if metric_fn is None:
        metric_fn = _sharpe_ratio

    T, N = M.shape
    if T < S:
        raise ValueError(f"T ({T}) must be >= S ({S})")

    values = M.values  # (T, N) numpy array
    blocks = _divide_into_blocks(T, S)

    # generate_combinatorial_splits returns (train_groups, test_groups) tuples
    # Here IS = S/2 blocks, OOS = S/2 blocks → we treat "test" as IS and
    # "train" as OOS (the symmetry doesn't matter; we iterate all C(S, S/2) combos).
    k = S // 2
    # Each element: (train_tuple, test_tuple) — we use test_tuple as IS
    splits = generate_combinatorial_splits(S, k)

    lambda_values = []
    for is_groups, oos_groups in splits:
        is_indices = np.concatenate([blocks[g] for g in is_groups])
        oos_indices = np.concatenate([blocks[g] for g in oos_groups])

        is_scores = np.array([metric_fn(values[is_indices, n]) for n in range(N)])
        oos_scores = np.array([metric_fn(values[oos_indices, n]) for n in range(N)])

        n_star = int(np.argmax(is_scores))
        oos_star = oos_scores[n_star]

        # Rank: number of strategies with strictly lower OOS score, normalised to (0,1)
        rank = float(np.sum(oos_scores < oos_star) + 1) / float(N + 1)
        # Clip to avoid log(0)
        rank = float(np.clip(rank, 1e-9, 1.0 - 1e-9))
        lambda_c = np.log(rank / (1.0 - rank))
        lambda_values.append(float(lambda_c))

    pbo = float(np.mean(np.array(lambda_values) < 0)) if lambda_values else float("nan")
    logger.info(
        "CSCV: S=%d, N=%d, combinations=%d, PBO=%.4f", S, N, len(lambda_values), pbo
    )
    return {"pbo": pbo, "lambda_values": lambda_values, "n_combinations": len(lambda_values)}


# ---------------------------------------------------------------------------
# Performance matrix builder
# ---------------------------------------------------------------------------

def build_performance_matrix(
    trials_oos: Dict[str, pd.DataFrame],
    mode: str = "auto",
    freq: str = "1h",
) -> pd.DataFrame:
    """
    Build performance matrix M (T × N) from per-trial OOS prediction DataFrames.

    Parameters
    ----------
    trials_oos : dict[str, pd.DataFrame]
        Mapping trial_id → DataFrame with columns ["y_true", "y_pred"] and
        a DatetimeIndex.
    mode : str
        "auto"          — detect per-trial: if y_true values look like prices
                          (all positive, large range) use "signal_return",
                          otherwise use "hit".
        "hit"           — per-step performance = 1 if y_pred == y_true else 0.
        "signal_return" — per-step performance = y_pred * pct_change(y_true).
    freq : str
        Resample frequency for temporal alignment (pandas offset string, e.g. "1h").

    Returns
    -------
    pd.DataFrame, shape (T, N)
        Inner-joined performance matrix aligned to common time index.
        Columns are the trial_ids from trials_oos.
    """
    if not trials_oos:
        return pd.DataFrame()

    series_dict: Dict[str, pd.Series] = {}
    for trial_id, df in trials_oos.items():
        if df.empty:
            continue
        y_true = df["y_true"].astype(float)
        y_pred = df["y_pred"].astype(float)

        effective_mode = mode
        if mode == "auto":
            unique_vals = np.unique(y_true.values)
            looks_like_prices = (
                len(unique_vals) > 10
                and float(y_true.min()) > 0
                and float(y_true.max()) > 10
            )
            effective_mode = "signal_return" if looks_like_prices else "hit"

        if effective_mode == "hit":
            perf = (y_pred == y_true).astype(float)
        else:  # signal_return
            ret = y_true.pct_change()
            perf = (y_pred.shift(1) * ret).fillna(0.0)

        # Resample to common frequency
        resampled = perf.resample(freq).mean()
        series_dict[trial_id] = resampled

    if not series_dict:
        return pd.DataFrame()

    M = pd.concat(series_dict, axis=1).dropna()
    M.columns.name = None
    return M
