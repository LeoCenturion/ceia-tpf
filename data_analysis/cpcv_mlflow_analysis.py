# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CPCV Strategy Comparison via MLflow
#
# Reads the parent-run metrics from each CPCV experiment and plots bar charts for:
# - Mean Sharpe Ratio across CPCV paths
# - Sharpe Ratio Variance across CPCV paths
# - Mean Probabilistic Sharpe Ratio (PSR) across CPCV paths

# %%
import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# %%
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

TRACKING_URI = f"sqlite:///{project_root}/mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)

EXPERIMENT_NAMES = [
    "CPCV_SmaCross",
    "CPCV_MaCrossover",
    "CPCV_BollingerBands",
    "CPCV_MACD",
    "CPCV_RSIDivergence",
    "CPCV_MultiIndicator",
]

# %% [markdown]
# ## Load parent-run metrics from MLflow

# %%
client = mlflow.tracking.MlflowClient()

records = []
for exp_name in EXPERIMENT_NAMES:
    experiment = client.get_experiment_by_name(exp_name)
    if experiment is None:
        print(f"[WARN] Experiment not found: {exp_name}")
        continue

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId = ''",  # parent runs only
        order_by=["start_time DESC"],
    )
    # Fall back to all top-level runs if tag filter returns nothing
    if not runs:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )
        runs = [r for r in runs if not r.data.tags.get("mlflow.parentRunId")]

    if not runs:
        print(f"[WARN] No parent runs found for: {exp_name}")
        continue

    # Use the most recent parent run
    run = runs[0]
    metrics = run.data.metrics
    records.append(
        {
            "strategy": exp_name.replace("CPCV_", ""),
            "sharpe_mean": metrics.get("sharpe_mean", float("nan")),
            "sharpe_variance": metrics.get("sharpe_std", float("nan")) ** 2,
            "psr_mean": metrics.get("psr_mean", float("nan")),
        }
    )

df = pd.DataFrame(records).set_index("strategy")
print(df.to_string())

# %% [markdown]
# ## Bar Charts

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("CPCV Strategy Comparison", fontsize=14, fontweight="bold")

bar_kwargs = dict(color="steelblue", edgecolor="white", width=0.6)

axes[0].bar(df.index, df["sharpe_mean"], **bar_kwargs)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_title("Mean Sharpe Ratio")
axes[0].set_ylabel("Sharpe Ratio")
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(df.index, df["sharpe_variance"], color="darkorange", edgecolor="white", width=0.6)
axes[1].set_title("Sharpe Ratio Variance")
axes[1].set_ylabel("Variance")
axes[1].tick_params(axis="x", rotation=30)

axes[2].bar(df.index, df["psr_mean"], color="seagreen", edgecolor="white", width=0.6)
axes[2].axhline(0.95, color="red", linewidth=0.8, linestyle="--", label="PSR = 0.95")
axes[2].set_ylim(0, 1)
axes[2].set_title("Mean Probabilistic Sharpe Ratio")
axes[2].set_ylabel("PSR")
axes[2].legend(fontsize=9)
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()

# %%
