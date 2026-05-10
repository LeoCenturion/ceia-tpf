"""
Backfill OOS predictions for existing MLflow trial runs.

For each Optuna trial in the target experiments, this script:
  1. Reconstructs the pipeline and model from the trial's logged params.
  2. Re-runs purged cross-validation with the exact same configuration.
  3. Logs the OOS predictions as a CSV artifact (predictions/oos_predictions.csv)
     to the existing MLflow trial run.

Skips trials that already have the predictions artifact, so it is safe to re-run.

Usage:
    python scripts/run_cscv_backfill.py
    python scripts/run_cscv_backfill.py --experiment Trading_Strategy_Optimization
    python scripts/run_cscv_backfill.py --limit 5         # process at most 5 trials per exp
    python scripts/run_cscv_backfill.py --dry-run          # log to test experiment instead
"""
from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — ensure repo root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_analysis.data_analysis import fetch_historical_data  # noqa: E402
from src.modeling.pipeline_runner import _log_oos_predictions  # noqa: E402
from src.modeling.machine_learning.rf_price_reversal import (  # noqa: E402
    RFClassifier,
    RFPriceReversalPipeline,
    RFTripleBarrierPipeline,
    RFTripleBarrierVolumePipeline,
)
from src.modeling.machine_learning.xgboost_pipeline_palazzo import (  # noqa: E402
    PalazzoXGBoostPipeline,
    XGBClassifierPalazzo,
)
from src.modeling.trading.trading_kfold_pipeline import (  # noqa: E402
    BollingerBandsClassifier,
    MACDClassifier,
    MaCrossoverClassifier,
    MultiIndicatorClassifier,
    RSIDivergenceClassifier,
    SmaCrossClassifier,
    TradingStrategyPipeline,
)
from src.modeling.statistical.statistical_pipelines import (  # noqa: E402
    ARIMAClassifier,
    ARIMAXGARCHClassifier,
    KalmanARIMAClassifier,
    SARIMAClassifier,
    SMAClassifier,
    StatisticalModelPipeline,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_URI = f"sqlite:///{_ROOT}/mlflow.db"
HOURLY_DATA_PATH = str(_ROOT / "data" / "BTCUSDT_1h.csv")
MINUTE_DATA_PATH = str(
    _ROOT
    / "data"
    / "binance"
    / "python"
    / "data"
    / "spot"
    / "daily"
    / "klines"
    / "BTCUSDT"
    / "1m"
    / "BTCUSDT_consolidated_klines.csv"
)

MODEL_CLS_MAP: dict[str, Any] = {
    # Trading
    "SmaCrossClassifier": SmaCrossClassifier,
    "MaCrossoverClassifier": MaCrossoverClassifier,
    "BollingerBandsClassifier": BollingerBandsClassifier,
    "MACDClassifier": MACDClassifier,
    "RSIDivergenceClassifier": RSIDivergenceClassifier,
    "MultiIndicatorClassifier": MultiIndicatorClassifier,
    # Statistical
    "SMAClassifier": SMAClassifier,
    "ARIMAClassifier": ARIMAClassifier,
    "SARIMAClassifier": SARIMAClassifier,
    "KalmanARIMAClassifier": KalmanARIMAClassifier,
    "ARIMAXGARCHClassifier": ARIMAXGARCHClassifier,
    # ML
    "RFClassifier": RFClassifier,
    "XGBClassifierPalazzo": XGBClassifierPalazzo,
}

PIPELINE_CLS_MAP: dict[str, Any] = {
    "TradingStrategyPipeline": TradingStrategyPipeline,
    "StatisticalModelPipeline": StatisticalModelPipeline,
    "RFPriceReversalPipeline": RFPriceReversalPipeline,
    "RFTripleBarrierPipeline": RFTripleBarrierPipeline,
    "RFTripleBarrierVolumePipeline": RFTripleBarrierVolumePipeline,
    "PalazzoXGBoostPipeline": PalazzoXGBoostPipeline,
}

# Default pipeline class per experiment (used when pipeline_class param is missing)
EXPERIMENT_DEFAULTS: dict[str, dict] = {
    "Trading_Strategy_Optimization": {
        "pipeline_cls": TradingStrategyPipeline,
        "data_path": MINUTE_DATA_PATH,
        "data_kwargs": {"symbol": "BTC/USDT", "timeframe": "1m"},
    },
    "Statistical_Models_Optimization": {
        "pipeline_cls": StatisticalModelPipeline,
        "data_path": HOURLY_DATA_PATH,
        "data_kwargs": {"start_date": "2022-01-01T00:00:00Z"},
    },
    "rf_price_reversal_optimization": {
        "pipeline_cls": RFPriceReversalPipeline,
        "data_path": HOURLY_DATA_PATH,
        "data_kwargs": {"start_date": "2022-01-01T00:00:00Z"},
    },
    "rf_triple_barrier_optimization": {
        "pipeline_cls": RFTripleBarrierPipeline,
        "data_path": HOURLY_DATA_PATH,
        "data_kwargs": {"start_date": "2022-01-01T00:00:00Z"},
    },
    "rf_volume_bars_optimization": {
        "pipeline_cls": RFTripleBarrierVolumePipeline,
        "data_path": MINUTE_DATA_PATH,
        "data_kwargs": {"symbol": "BTC/USDT", "timeframe": "1m"},
    },
    "Palazzo_XGBoost_Optimization": {
        "pipeline_cls": PalazzoXGBoostPipeline,
        "data_path": MINUTE_DATA_PATH,
        "data_kwargs": {"symbol": "BTC/USDT", "timeframe": "1m"},
    },
}

TARGET_EXPERIMENTS = list(EXPERIMENT_DEFAULTS.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce(value: str) -> Any:
    """Coerce a MLflow string param to its most specific Python type."""
    for typ in (int, float):
        try:
            return typ(value)
        except (ValueError, TypeError):
            pass
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _extract_params(parent_run, trial_run) -> tuple[dict, dict, str, str]:
    """
    Extract (pipeline_config, model_params, model_class_name, pipeline_class_name)
    from a parent + trial run pair.

    Handles two logging formats:
      - New (RF/XGBoost): trial has pipeline_config.* and model_params.* prefixes.
      - Old (Statistical/Trading): trial has flat model params; parent has pipeline.*
        and model_class.
    """
    parent_params = {k: _coerce(v) for k, v in parent_run.data.params.items()}
    trial_params = {k: _coerce(v) for k, v in trial_run.data.params.items()}

    pipeline_config: dict = {}
    model_params: dict = {}

    # Metadata keys to skip when collecting flat model params
    _SKIP_KEYS = {
        "model_class", "pipeline_class", "n_trials", "pipeline_config",
        "data_filename", "data_start_date", "data_end_date", "data_rows", "data_cols",
    }

    def _collect(params: dict, from_parent: bool = False) -> None:
        for k, v in params.items():
            if k.startswith("pipeline_config."):
                pipeline_config[k[len("pipeline_config."):]] = v
            elif k.startswith("pipeline."):
                pipeline_config[k[len("pipeline."):]] = v
            elif k.startswith("model_params."):
                model_params[k[len("model_params."):]] = v
            elif not from_parent and k not in _SKIP_KEYS and not k.startswith("best."):
                # Flat trial param → model param (statistical/trading format)
                model_params[k] = v

    _collect(parent_params, from_parent=True)
    _collect(trial_params, from_parent=False)

    model_class_name = str(
        trial_params.get("model_class", parent_params.get("model_class", ""))
    )
    pipeline_class_name = str(
        trial_params.get("pipeline_class", parent_params.get("pipeline_class", ""))
    )
    return pipeline_config, model_params, model_class_name, pipeline_class_name


def _artifact_exists(client: mlflow.tracking.MlflowClient, run_id: str) -> bool:
    """Return True if the predictions artifact already exists for this run."""
    try:
        artifacts = client.list_artifacts(run_id, path="predictions")
        return any("oos_predictions" in a.path for a in artifacts)
    except Exception:
        return False


def _load_data(experiment_name: str, pipeline_config: dict) -> pd.DataFrame:
    """Load raw data for the given experiment."""
    defaults = EXPERIMENT_DEFAULTS.get(experiment_name, {})
    # Prefer data_path stored in pipeline_config over the hardcoded default
    data_path = pipeline_config.get("data_path") or defaults.get("data_path", HOURLY_DATA_PATH)
    kwargs = dict(defaults.get("data_kwargs", {}))
    logger.info("Loading data from %s", data_path)
    return fetch_historical_data(data_path=data_path, **kwargs)


# ---------------------------------------------------------------------------
# Core backfill logic
# ---------------------------------------------------------------------------

def backfill_experiment(
    experiment_name: str,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """
    Process all trial runs in *experiment_name*, patch each with OOS predictions.
    Returns the number of trials successfully patched.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient(MLFLOW_URI)

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.warning("Experiment %r not found — skipping.", experiment_name)
        return 0

    # Find all parent runs (they have model_class logged)
    parent_runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.`mlflow.runName` LIKE '%_optuna'",
    )
    if not parent_runs:
        logger.warning("No parent optuna runs found in %r.", experiment_name)
        return 0

    defaults = EXPERIMENT_DEFAULTS.get(experiment_name, {})
    default_pipeline_cls = defaults.get("pipeline_cls")

    raw_data_cache: dict[str, pd.DataFrame] = {}

    patched = 0
    for parent_run in parent_runs:
        parent_id = parent_run.info.run_id

        trial_runs = client.search_runs(
            [exp.experiment_id],
            filter_string=(
                f"tags.`mlflow.parentRunId` = '{parent_id}' "
                "AND tags.`mlflow.runName` LIKE 'trial_%'"
            ),
            order_by=["attributes.start_time ASC"],
        )
        logger.info(
            "[%s] Parent: %s → %d trial runs",
            experiment_name,
            parent_run.info.run_name,
            len(trial_runs),
        )

        processed = 0
        for trial_run in trial_runs:
            if limit is not None and processed >= limit:
                break
            if _artifact_exists(client, trial_run.info.run_id):
                logger.info("  trial %s: already has predictions, skipping.", trial_run.info.run_name)
                continue

            pipeline_config, model_params, model_class_name, pipeline_class_name = (
                _extract_params(parent_run, trial_run)
            )

            pipeline_cls = PIPELINE_CLS_MAP.get(pipeline_class_name) or default_pipeline_cls
            model_cls = MODEL_CLS_MAP.get(model_class_name)

            if pipeline_cls is None or model_cls is None:
                logger.warning(
                    "  Cannot resolve pipeline=%r model=%r — skipping trial %s.",
                    pipeline_class_name,
                    model_class_name,
                    trial_run.info.run_name,
                )
                continue

            # Load data (cache per experiment to avoid repeated IO)
            cache_key = experiment_name
            if cache_key not in raw_data_cache:
                try:
                    raw_data_cache[cache_key] = _load_data(experiment_name, pipeline_config)
                except Exception as exc:
                    logger.error("  Failed to load data for %r: %s", experiment_name, exc)
                    break
            raw_data = raw_data_cache[cache_key]

            try:
                model = model_cls(**model_params)
                pipeline = pipeline_cls(pipeline_config)
                *_, oos_df = pipeline.run_cv(raw_data, model)
            except Exception as exc:
                logger.warning(
                    "  trial %s failed during run_cv: %s",
                    trial_run.info.run_name,
                    exc,
                )
                continue

            if oos_df is None or oos_df.empty:
                logger.warning("  trial %s: run_cv produced empty OOS predictions.", trial_run.info.run_name)
                continue

            target_run_id = trial_run.info.run_id
            if dry_run:
                target_exp = f"{experiment_name}_backfill_dryrun"
                mlflow.set_experiment(target_exp)
                with mlflow.start_run(run_name=f"dryrun_{trial_run.info.run_name}"):
                    _log_oos_predictions(oos_df)
                logger.info(
                    "  [DRY RUN] trial %s: logged %d rows to %s.",
                    trial_run.info.run_name,
                    len(oos_df),
                    target_exp,
                )
            else:
                with mlflow.start_run(run_id=target_run_id):
                    _log_oos_predictions(oos_df)
                logger.info(
                    "  trial %s: patched with %d OOS predictions.",
                    trial_run.info.run_name,
                    len(oos_df),
                )
            patched += 1
            processed += 1

    return patched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill OOS predictions for existing MLflow trials.")
    parser.add_argument(
        "--experiment",
        nargs="+",
        default=TARGET_EXPERIMENTS,
        help="Experiment name(s) to process (default: all target experiments).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of trials to patch per parent run (useful for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log predictions to a test experiment instead of patching existing runs.",
    )
    args = parser.parse_args()

    total = 0
    for exp_name in args.experiment:
        logger.info("=== Processing experiment: %s ===", exp_name)
        n = backfill_experiment(exp_name, limit=args.limit, dry_run=args.dry_run)
        logger.info("=== %s: patched %d trials ===", exp_name, n)
        total += n

    logger.info("Done. Total trials patched: %d", total)


if __name__ == "__main__":
    main()
