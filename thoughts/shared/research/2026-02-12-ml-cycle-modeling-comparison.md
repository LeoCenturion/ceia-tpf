---
date: 2026-02-12T10:00:00-03:00
researcher: Gemini CLI
git_commit: 720bc2c8713b5f4bbf130404dd1878c9157dd6f8
branch: master
repository: Tp Final
topic: "Machine Learning Cycle Analysis: Palazzo vs. Statistical Methods"
tags: [research, codebase, xgboost, random-forest, labeling, stationarity]
status: complete
last_updated: 2026-02-12
last_updated_by: Gemini CLI
---

# Research: Machine Learning Cycle Analysis

**Date**: 2026-02-12
**Researcher**: Gemini CLI
**Git Commit**: 720bc2c8713b5f4bbf130404dd1878c9157dd6f8
**Branch**: master
**Repository**: Tp Final

## Research Question
Research the machine learning cycle phases of `src/modeling/xgboost_price_reversal_palazzo.py` (Palazzo implementation) and compare it with the framework in `src/modeling/statistical_methods.py`. Focus on data curation, feature engineering, labeling, and modeling.

## Summary
The codebase contains two distinct philosophies for financial machine learning pipelines. The **Palazzo implementation** (`xgboost_price_reversal_palazzo.py`) focuses on GPU-accelerated gradient boosting with a dynamic, volatility-aware labeling method and walk-forward validation. In contrast, the **Statistical Methods implementation** (`statistical_methods.py`) follows the "Advances in Financial Machine Learning" (De Prado) framework, emphasizing stationarity with memory preservation (FFD), information-driven bars (Dollar Bars), and rigorous leakage prevention (PurgedKFold).

## Detailed Findings

### 1. Data Curation & Bar Aggregation
- **Palazzo (`xgboost_...`)**:
    - **Aggregation**: Uses **Volume Bars** (`aggregate_to_volume_bars`).
    - **Intra-bar Metrics**: Calculates `intra_bar_std` (standard deviation of log-returns within the bar). This is a critical metric used later in the labeling phase.
    - **Philosopy**: Information is sampled based on a volume threshold, and volatility is measured locally within those samples.
- **Statistical Methods (`statistical_...`)**:
    - **Aggregation**: Uses **Dollar Bars** (`create_dollar_bars`).
    - **Philosophy**: Samples data based on fiat value exchanged, which is more stable across price regimes. It focuses on handling noise through sample uniqueness rather than intra-bar metrics.

### 2. Feature Engineering & Stationarity
- **Palazzo**:
    - **Stationarity**: Uses standard **Percentage Changes** (`pct_change()`). This is equivalent to integer differentiation ($d=1$), which ensures stationarity but eliminates the "memory" (autocorrelation) of the price series.
    - **Features**: Includes a wide array of technical indicators (RSI, Stoch, MACD) and lagged returns.
- **Statistical Methods**:
    - **Stationarity**: Employs **Fractional Differentiation (FFD)** via `find_minimum_d`. This searches for the minimum $d \in [0, 1]$ that achieves stationarity (ADF test), preserving maximum memory.
    - **Orthogonalization**: Uses **PCA** to transform stationary features into orthogonal components, preventing multicollinearity issues common in technical indicators.

### 3. Labeling & Sample Weighting
- **Palazzo**:
    - **Target**: A binary "top" reversal label. It uses a **dynamic horizontal threshold**: 
      `next_bar_return >= (bar_return + intra_bar_std * tau)`.
    - **Weighting**: Uses **Balanced Class Weights** (`compute_class_weight`) to address the rarity of reversal points during training.
- **Statistical Methods**:
    - **Target**: Standard **Triple-Barrier Method (TBM)** with profit-taking, stop-loss, and time-exhaustion barriers.
    - **Weighting**: Uses **Sample Uniqueness** and **Volatility-based Weighting**. Uniqueness corrects for serial correlation in overlapping labels, while volatility weighting emphasizes high-magnitude moves.

### 4. Modeling & Validation
- **Palazzo**:
    - **Model**: **XGBoost** with GPU acceleration (`device='cuda'`).
    - **Validation**: **Walk-Forward Backtest** (`manual_backtest`) with periodic refitting.
    - **Optimization**: Integrated with **Optuna** to tune both model parameters and data aggregation parameters (volume threshold, tau).
- **Statistical Methods**:
    - **Model**: **Random Forest** (usually).
    - **Validation**: **PurgedKFold** with **Embargo**. This rigorously prevents data leakage by "purging" overlapping training samples and "embargoing" samples immediately after the test set.

## Code References
- `src/modeling/xgboost_price_reversal_palazzo.py:270` - `aggregate_to_volume_bars` implementation.
- `src/modeling/xgboost_price_reversal_palazzo.py:326` - Dynamic reversal labeling logic.
- `src/modeling/statistical_methods.py:126` - Fractional Differentiation (FFD) integration.
- `src/modeling/statistical_methods.py:217` - Triple-Barrier Method labeling.
- `src/modeling/__init__.py` - `PurgedKFold` implementation for leakage prevention.

## Architecture Insights
- The Palazzo approach is more **experimental and optimization-driven**, treating the data generation process as a hyperparameter.
- The Statistical Methods approach is more **structural and econometric**, prioritizing the validity of the statistical properties (stationarity, independence) over raw predictive power in a backtest.
- **Leakage Prevention**: The `statistical_methods.py` pipeline is significantly more robust against serial correlation leakage due to its use of PurgedKFold and Embargo.

## Related Research
- `thoughts/shared/research/2026-02-12-ml-cycle-modeling-comparison.md` (This document)

## Open Questions
- How does the performance of Palazzo's dynamic intra-bar volatility threshold compare against the standard TBM in highly volatile regimes?
- Could the FFD logic from `statistical_methods.py` be integrated into the `xgboost_...` pipeline to improve memory preservation while maintaining XGBoost's speed?
