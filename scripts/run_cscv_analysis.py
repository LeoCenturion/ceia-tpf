"""
CSCV (Combinatorially Symmetric Cross-Validation) analysis for MLflow experiments.

Loads OOS predictions from all trial runs in a given experiment, builds the
performance matrix, and computes PBO (Probability of Backtest Overfitting).

Performance series are cached to disk (.cache/cscv/) so re-runs skip downloads
and avoid re-computing. Delete the cache dir to force a refresh.

Usage:
    python scripts/run_cscv_analysis.py
    python scripts/run_cscv_analysis.py --experiment Trading_Strategy_Optimization
    python scripts/run_cscv_analysis.py --experiment Trading_Strategy_Optimization --S 16
    python scripts/run_cscv_analysis.py --mode hit   # for classifiers
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.backtesting.cscv import compute_pbo  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_URI = f"sqlite:///{_ROOT}/mlflow.db"
CACHE_DIR = _ROOT / ".cache" / "cscv"

# Default mode per experiment — trading stores close prices, classifiers store labels
EXPERIMENT_MODES: dict[str, str] = {
    "Trading_Strategy_Optimization": "signal_return",
    "Statistical_Models_Optimization": "hit",
    "rf_price_reversal_optimization": "hit",
    "rf_triple_barrier_optimization": "hit",
    "rf_volume_bars_optimization": "hit",
    "Palazzo_XGBoost_Optimization": "hit",
}


def _perf_series(df: pd.DataFrame, mode: str, freq: str) -> pd.Series:
    """Convert a (y_true, y_pred) DataFrame to a resampled performance Series."""
    y_true = df["y_true"].astype(float)
    y_pred = df["y_pred"].astype(float)
    if mode == "hit":
        perf = (y_pred == y_true).astype(float)
    else:  # signal_return
        ret = y_true.pct_change()
        perf = (y_pred.shift(1) * ret).fillna(0.0)
    return perf.resample(freq).mean()


def load_performance_matrix(
    experiment_name: str,
    mode: str,
    freq: str,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """
    Build the T×N performance matrix for CSCV.

    For each trial run:
      1. Check disk cache (.cache/cscv/{run_id}_{mode}_{freq}.parquet)
      2. If missing, download the OOS CSV from MLflow, compute the performance
         series, persist to cache, then discard the raw DataFrame immediately.
      3. Assemble all series into a matrix and inner-join on the common time index.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment {experiment_name!r} not found.")

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.`mlflow.runName` LIKE 'trial_%'",
    )
    logger.info("Found %d trial runs in %s", len(runs), experiment_name)

    safe_freq = freq.replace("/", "_")
    series_dict: dict[str, pd.Series] = {}
    downloaded = cached = skipped = 0

    for run in runs:
        run_id = run.info.run_id
        cache_file = cache_dir / f"{run_id}_{mode}_{safe_freq}.parquet"

        if cache_file.exists():
            perf = pd.read_parquet(cache_file)["perf"]
            cached += 1
        else:
            artifacts = client.list_artifacts(run_id, path="predictions")
            pred_artifacts = [a for a in artifacts if "oos_predictions" in a.path]
            if not pred_artifacts:
                skipped += 1
                continue

            artifact_path = pred_artifacts[0].path
            # Artifacts are stored locally under artifact_uri — read directly
            # to avoid MLflow 3.x download_artifacts path-prefix bugs.
            local_path = Path(run.info.artifact_uri) / artifact_path
            if not local_path.exists():
                logger.warning("  %s: artifact not found at %s, skipping.", run.info.run_name, local_path)
                skipped += 1
                continue

            df = pd.read_csv(local_path, index_col=0, parse_dates=True)
            perf = _perf_series(df, mode, freq)
            del df  # free raw data immediately

            perf.to_frame("perf").to_parquet(cache_file)
            downloaded += 1

        series_dict[run.info.run_name] = perf

    logger.info(
        "Series loaded — cached: %d, downloaded: %d, skipped (no artifact): %d",
        cached, downloaded, skipped,
    )

    if not series_dict:
        return pd.DataFrame()

    M = pd.concat(series_dict, axis=1).dropna()
    M.columns.name = None
    return M


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CSCV/PBO for an MLflow experiment.")
    parser.add_argument(
        "--experiment",
        default="Trading_Strategy_Optimization",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--S",
        type=int,
        default=8,
        help="Number of CSCV blocks (must be even). S=8 → 70 combinations, S=16 → 12870.",
    )
    parser.add_argument(
        "--freq",
        default="1h",
        help="Resample frequency for alignment (pandas offset string).",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["auto", "hit", "signal_return"],
        help="Performance metric mode. Defaults to per-experiment setting.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)

    mode = args.mode or EXPERIMENT_MODES.get(args.experiment, "auto")
    M = load_performance_matrix(args.experiment, mode=mode, freq=args.freq)

    if M.empty:
        logger.error("Performance matrix is empty — check that trial OOS dates overlap.")
        sys.exit(1)

    logger.info("Performance matrix: %d time steps × %d strategies", *M.shape)

    if M.shape[0] < args.S:
        logger.error(
            "Not enough time steps (%d) for S=%d. Lower --S or use a coarser --freq.",
            M.shape[0], args.S,
        )
        sys.exit(1)

    logger.info("Computing PBO (S=%d)…", args.S)
    result = compute_pbo(M, S=args.S)

    print(f"\n=== CSCV Results: {args.experiment} ===")
    print(f"Trials (N):        {M.shape[1]}")
    print(f"Time steps (T):    {M.shape[0]}")
    print(f"Combinations:      {result['n_combinations']}")
    print(f"PBO:               {result['pbo']:.4f}")
    print()
    pbo = result["pbo"]
    if pbo < 0.1:
        print("Interpretation: Low overfitting risk — IS winner tends to win OOS.")
    elif pbo < 0.4:
        print("Interpretation: Moderate overfitting risk.")
    else:
        print("Interpretation: High overfitting risk — strategy selection is likely overfit.")


if __name__ == "__main__":
    main()
