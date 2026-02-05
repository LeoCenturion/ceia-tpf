# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 1

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.statistical_methods import (
    machine_learning_cycle,
    feature_importance_mdi,
    feature_importance_mda,
    feature_importance_sfi,
    feature_importance_orthogonal,
    weighted_kendalls_tau
)
from src.modeling import PurgedKFold

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
# # Feature Importance Analysis
# 
# This notebook executes the machine learning pipeline and analyzes feature importance using four distinct methods:
# 1. **MDI (Mean Decrease Impurity):** Standard Random Forest importance (on Orthogonal Features).
# 2. **MDA (Mean Decrease Accuracy):** Permutation importance (on Orthogonal Features).
# 3. **SFI (Single Feature Importance):** Predictive power of features in isolation (on Original Features).
# 4. **Orthogonal Importance:** Importance of PCA components vs. their explained variance.
# 
# Finally, we verify the results using Weighted Kendall's Tau.

# %%
# --- Configuration ---
config = {
    "dollar_threshold": 5_000_000,
    "horizon": 5,
    "pt": 1,
    "sl": 1,
    "min_ret": 0.0005,
    "n_splits": 3,
    "pct_embargo": 0.01,
}

# --- Data Loading ---
data_path = os.path.join(project_root, "data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv")

raw_tick_data = fetch_historical_data(
    symbol="BTC/USDT",
    timeframe="1m",
    start_date="2025-06-01T00:00:00Z",
    end_date="2025-08-01T00:00:00Z",
    data_path=data_path,
)
raw_tick_data.index = pd.to_datetime(raw_tick_data.index)

print(f"Data loaded: {raw_tick_data.shape}")

# %%
# --- Pipeline Execution ---
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Running ML Pipeline... (This may take a few minutes)")
trained_model, scores, X, y, sample_weights, t1, features, pca = machine_learning_cycle(raw_tick_data, model, config)

print(f"\nCross-validation F1 Scores: {scores}")
print(f"Average F1 Score: {np.mean(scores):.4f}")

# %%
# ## 1. Mean Decrease Impurity (MDI)
# MDI measures how much each feature decreases impurity (Gini) in the trees. Here, applied to the **Orthogonal Features (Principal Components)**.

# %%
mdi_importance = feature_importance_mdi(trained_model, X, y)

plt.figure(figsize=(15, 6))
mdi_importance.head(20).plot(kind='bar', color='skyblue')
plt.title("Top 20 Features - Mean Decrease Impurity (MDI)")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# ## 2. Mean Decrease Accuracy (MDA)
# MDA measures how much the model's performance drops when a feature is randomly shuffled. Applied to **Orthogonal Features**.

# %%
cv = PurgedKFold(
    n_splits=config["n_splits"],
    t1=t1,
    pct_embargo=config["pct_embargo"],
)

mda_importance = feature_importance_mda(trained_model, X, y, cv, sample_weights, t1)

# Remove 'full_model' and filter out features with non-positive importance (harmful or irrelevant)
plot_mda = mda_importance.drop('full_model', errors='ignore')
plot_mda = plot_mda[plot_mda > 0]
# Sorting is already handled by the library, but we re-sort to ensure descending order after filtering
plot_mda = plot_mda.sort_values(ascending=False)

plt.figure(figsize=(15, 6))
plot_mda.head(20).plot(kind='bar', color='salmon')
plt.title("Top 20 Features - Mean Decrease Accuracy (MDA)")
plt.ylabel("Importance (Loss Increase / Accuracy Drop)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# ## 3. Single Feature Importance (SFI)
# SFI trains a separate model for each feature. This is applied to the **Original Features**, making it highly interpretable for identifying the best raw technical indicators.

# %%
# We align original features with the processed target index
features_aligned = features.loc[X.index]

sfi_importance = feature_importance_sfi(
    trained_model,
    features_aligned,
    y,
    cv,
    sample_weights,
    t1,
)

plt.figure(figsize=(15, 6))
sfi_importance.head(20).plot(kind='bar', color='lightgreen')
plt.title("Top 20 Features - Single Feature Importance (SFI)")
plt.ylabel("F1 Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# ## 4. Orthogonal Feature Importance & Kendall's Tau
# Comparing the ML importance of Principal Components against their Explained Variance.

# %%
ortho_importance = feature_importance_orthogonal(
    trained_model,
    X,
    y,
    sample_weights,
    pca,
)

# Calculate Correlation
ml_importance = ortho_importance["Orthogonal Importance"]
eigen_importance = ortho_importance["Explained Variance"]
tau, p_value = weighted_kendalls_tau(ml_importance, eigen_importance)

print(f"Weighted Kendall's Tau Correlation: {tau:.4f}")
if not np.isnan(p_value):
    print(f"p-value: {p_value:.4f}")

# Plot
fig, ax1 = plt.subplots(figsize=(15, 6))

color = 'tab:blue'
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('ML Importance', color=color)
ax1.bar(ortho_importance.index[:20], ortho_importance["Orthogonal Importance"][:20], color=color, alpha=0.6, label='ML Importance')
ax1.tick_params(axis='y', labelcolor=color)
plt.xticks(rotation=45, ha='right')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Explained Variance', color=color)
ax2.plot(ortho_importance.index[:20], ortho_importance["Explained Variance"][:20], color=color, marker='o', label='Explained Variance')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("ML Importance vs Explained Variance (Top 20 PCs)")
fig.tight_layout()
plt.show()

# %%
# ## 5. Autocorrelation Analysis of Top Features
# We analyze the serial correlation of the top relevant features identified by SFI (Original Features).

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

top_n_features = 5
top_features = sfi_importance.head(top_n_features).index

print(f"\n--- Top {top_n_features} SFI Features for ACF/PACF Analysis ---")
print(top_features.tolist())

for feature_name in top_features:
    feature_data = features_aligned[feature_name].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    fig.suptitle(f'Autocorrelation Analysis: {feature_name}', fontsize=14)
    
    plot_acf(feature_data, lags=40, ax=axes[0], title=f'ACF - {feature_name}')
    plot_pacf(feature_data, lags=40, ax=axes[1], title=f'PACF - {feature_name}')
    
    plt.tight_layout()
    plt.show()
