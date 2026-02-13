# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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

# %%
# %load_ext autoreload
# %autoreload 2
# %reload_ext autoreload
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
def plot_importance(importance_series, title, color="skyblue"):
    """
    Helper function to plot feature importance.
    """
    if importance_series.empty:
        print(f"No positive importance found for {title}")
        return
        
    plt.figure(figsize=(15, 6))
    importance_series.head(20).plot(kind='bar', color=color)
    plt.title(title)
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(data_dict, title):
    """
    Helper function to compute and plot a pairwise Weighted Kendall's Tau heatmap.
    
    Args:
        data_dict (dict): Dictionary where keys are names and values are pd.Series of importance scores.
        title (str): Title for the heatmap.
    """
    # Create a DataFrame from the series, aligning indices (inner join)
    df = pd.DataFrame(data_dict).dropna()
    
    if df.empty:
        print(f"No overlapping features found for correlation heatmap: {title}")
        return

    cols = df.columns
    n = len(cols)
    corr_matrix = np.zeros((n, n))
    
    print(f"\nComputing Weighted Kendall's Tau Correlation Matrix for: {list(cols)}")
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # weighted_kendalls_tau handles alignment internally, but we already aligned in df
                tau, _ = weighted_kendalls_tau(df.iloc[:, i], df.iloc[:, j])
                corr_matrix[i, j] = tau
                
    corr_df = pd.DataFrame(corr_matrix, index=cols, columns=cols)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_original_features(features, y, cv, sample_weights, t1, model_params):
    """
    Computes and plots MDI, MDA, and SFI for original features.
    """
    print("\n--- Analyzing Original Features ---")
    
    # Align features with target
    X = features.loc[y.index]
    
    # 1. MDI (Requires fitting a model on original features)
    print("Computing MDI for Original Features...")
    model = RandomForestClassifier(**model_params)
    model.fit(X, y, sample_weight=sample_weights)
    mdi = feature_importance_mdi(model, X, y)
    plot_importance(mdi, "Original Features - MDI", "skyblue")
    
    # 2. MDA
    print("Computing MDA for Original Features...")
    model_mda = RandomForestClassifier(**model_params)
    mda = feature_importance_mda(model_mda, X, y, cv, sample_weights, t1)
    # Invert log-loss: we want (Loss_perm - Loss_orig)
    mda = -mda 
    
    # Filter for plotting (positive only) but keep raw for correlation analysis
    plot_mda = mda.drop('full_model', errors='ignore')
    plot_mda = plot_mda[plot_mda > 0].sort_values(ascending=False)
    plot_importance(plot_mda, "Original Features - MDA", "salmon")
    
    # 3. SFI
    print("Computing SFI for Original Features...")
    model_sfi = RandomForestClassifier(**model_params)
    sfi = feature_importance_sfi(model_sfi, X, y, cv, sample_weights, t1)
    plot_importance(sfi.sort_values(ascending=False), "Original Features - SFI", "lightgreen")
    
    # 4. Correlation Heatmap
    # Use raw scores (including negatives) for correlation to capture the full relationship
    mda_clean = mda.drop('full_model', errors='ignore')
    
    plot_correlation_heatmap(
        {
            "MDI": mdi,
            "MDA": mda_clean,
            "SFI": sfi
        },
        "Correlation of Feature Importance Metrics (Original Features)"
    )
    
    return mdi, mda, sfi


def plot_pca_features(X_pca, y, cv, sample_weights, t1, trained_model):
    """
    Computes and plots MDI, MDA, and SFI for PCA features.
    """
    print("\n--- Analyzing PCA Features ---")
    
    # 1. MDI (Use trained_model which is already fitted on X_pca)
    print("Computing MDI for PCA Features...")
    mdi = feature_importance_mdi(trained_model, X_pca, y)
    plot_importance(mdi, "PCA Features - MDI", "skyblue")
    
    # 2. MDA
    print("Computing MDA for PCA Features...")
    # Use clone to ensure fresh model for MDA CV
    mda = feature_importance_mda(clone(trained_model), X_pca, y, cv, sample_weights, t1)
    mda = -mda
    
    # Filter for plotting
    plot_mda = mda.drop('full_model', errors='ignore')
    plot_mda = plot_mda[plot_mda > 0].sort_values(ascending=False)
    plot_importance(plot_mda, "PCA Features - MDA", "salmon")
    
    # 3. SFI (Run on PCA features)
    print("Computing SFI for PCA Features...")
    sfi = feature_importance_sfi(trained_model, X_pca, y, cv, sample_weights, t1)
    plot_importance(sfi.sort_values(ascending=False), "PCA Features - SFI", "lightgreen")
    
    return mdi, mda, sfi


def compare_pca_importance(mdi_ortho, mda_ortho, sfi_ortho, pca, X_pca):
    """
    Computes Weighted Kendall's Tau for MDI, MDA, SFI vs PCA Explained Variance.
    """
    print("\n--- Comparing ML Importance vs PCA Explained Variance ---")
    
    explained_variance = pd.Series(
        pca.explained_variance_ratio_, index=X_pca.columns, name="Explained Variance"
    )
    
    # Drop full_model from MDA if present
    mda_clean = mda_ortho.drop('full_model', errors='ignore') if isinstance(mda_ortho, pd.Series) else mda_ortho
    
    # Plot Correlation Heatmap including Explained Variance
    plot_correlation_heatmap(
        {
            "MDI (Ortho)": mdi_ortho,
            "MDA (Ortho)": mda_clean,
            "SFI (Ortho)": sfi_ortho,
            "PCA Var": explained_variance
        },
        "Correlation of ML Importance vs PCA Variance"
    )
    
    # Plot Comparison for MDI (Standard Orthogonal Importance)
    results = pd.concat([mdi_ortho.rename("Orthogonal Importance"), explained_variance], axis=1)
    results = results.sort_values(by="Orthogonal Importance", ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(15, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('MDI Importance', color=color)
    ax1.bar(results.index[:20], results["Orthogonal Importance"][:20], color=color, alpha=0.6, label='MDI Importance')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xticks(rotation=45, ha='right')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Explained Variance', color=color)
    ax2.plot(results.index[:20], results["Explained Variance"][:20], color=color, marker='o', label='Explained Variance')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("MDI Importance vs Explained Variance (Top 20 PCs)")
    fig.tight_layout()
    plt.show()


# %%
# --- Configuration ---
config = {
    "dollar_threshold": 1e9,
    "horizon": 5,
    "pt": 1,
    "sl": 1,
    "min_ret": 0.0005,
    "n_splits": 3,
    "pct_embargo": 0.01,
    "feature_whitelist": None,
    "pca_whitelist": None
}

# --- Data Loading ---
data_path = os.path.join(project_root, "data/binance/python/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT_consolidated_klines.csv")

raw_tick_data = fetch_historical_data(
    symbol="BTC/USDT",
    timeframe="1h",
    start_date="2020-01-01T00:00:00Z",
    end_date="2025-08-01T00:00:00Z",
    data_path=data_path,
)
raw_tick_data.index = pd.to_datetime(raw_tick_data.index)

print(f"Data loaded: {raw_tick_data.shape}")

# %%
# --- Pipeline Execution ---
model_params = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
model = RandomForestClassifier(**model_params)

print("Running ML Pipeline... (This may take a few minutes)")
trained_model, scores, X, y, sample_weights, t1, features, pca = machine_learning_cycle(raw_tick_data, model, config)

print(f"\nCross-validation F1 Scores: {scores}")
print(f"Average F1 Score: {np.mean(scores):.4f}")

# %%
# Define CV for feature importance
cv = PurgedKFold(
    n_splits=config["n_splits"],
    t1=t1,
    pct_embargo=config["pct_embargo"],
)

# 1. Analyze Original Features
mdi_orig, mda_orig, sfi_orig = plot_original_features(
    features, y, cv, sample_weights, t1, model_params
)

# 2. Analyze PCA Features
mdi_ortho, mda_ortho, sfi_ortho = plot_pca_features(
    X, y, cv, sample_weights, t1, trained_model
)

# 3. Compare PCA Importance
compare_pca_importance(mdi_ortho, mda_ortho, sfi_ortho, pca, X)

# %%
# ## 4. Autocorrelation Analysis of Top Features
# We analyze the serial correlation of the top relevant features identified by SFI (Original Features).

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Use the top SFI features from the original feature set
# sfi_orig is a Series with index=feature_name, values=score
top_n_features = 5
top_features = sfi_orig.sort_values(ascending=False).head(top_n_features).index

print(f"\n--- Top {top_n_features} SFI Features for ACF/PACF Analysis ---")
print(top_features.tolist())

features_aligned = features.loc[X.index]

for feature_name in top_features:
    feature_data = features_aligned[feature_name].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    fig.suptitle(f'Autocorrelation Analysis: {feature_name}', fontsize=14)
    
    plot_acf(feature_data, lags=40, ax=axes[0], title=f'ACF - {feature_name}')
    plot_pacf(feature_data, lags=40, ax=axes[1], title=f'PACF - {feature_name}')
    
    plt.tight_layout()
    plt.show()
