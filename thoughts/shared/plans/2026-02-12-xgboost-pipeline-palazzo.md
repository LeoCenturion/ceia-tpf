# Implementation Plan - Palazzo XGBoost Pipeline with Purged CV

## Overview

We will create a new modeling pipeline in `src/modeling/xgboost_pipeline_palazzo.py`. This pipeline will adopt the structured, class-based architecture of `MachineLearningPipeline` from `src/modeling/statistical_methods.py` but will utilize the specific data processing and modeling logic from `src/modeling/xgboost_price_reversal_palazzo.py`.

Crucially, we will retain the robust **Purged Cross-Validation** (PurgedKFold) from the statistical methods pipeline to ensure rigorous backtesting without data leakage, applying it to the Palazzo strategy's volume bars and labeling.

## Current State Analysis

- **Source A (`statistical_methods.py`)**: A structured pipeline using Dollar Bars, FFD stationarity, Triple-Barrier labeling, and PurgedKFold CV. It's architecturally sound but uses different math.
- **Source B (`xgboost_price_reversal_palazzo.py`)**: A script using Volume Bars, Percentage Change stationarity, Dynamic "Top" labeling, and XGBoost with a simple Walk-Forward backtest. It's the target strategy.
- **Goal**: Merge the *Strategy* of B into the *Architecture* of A.

## Desired End State

- A new file `src/modeling/xgboost_pipeline_palazzo.py` containing `PalazzoXGBoostPipeline`.
- The pipeline processes raw data into Volume Bars.
- It generates features using Palazzo's technical indicators (RSI, MACD, etc.) and percentage changes.
- It labels data using the dynamic volatility threshold.
- It trains an XGBoost classifier using **PurgedKFold Cross-Validation**.
- It outputs performance metrics (F1 score) and feature importance.

## Key Discoveries
- **Labeling Mismatch**: Palazzo's labeling looks 1 bar ahead (`shift(-1)`). To use `PurgedKFold`, we must explicitly define the `t1` (event end time) series as the timestamp of the *next* bar.
- **Weighting**: Palazzo uses class weights (balanced) rather than uniqueness-based weights. We will implement class weighting but pass it as sample weights to the model to fit the pipeline's interface.
- **PCA**: Palazzo's XGBoost model operates on raw technical features. We will **remove** the PCA step present in `statistical_methods.py` to faithfully reproduce the Palazzo model's behavior.

## What We're NOT Doing
- We are NOT keeping Fractional Differentiation (FFD). We use Palazzo's `pct_change`.
- We are NOT keeping Triple-Barrier Method. We use Palazzo's dynamic threshold.
- We are NOT keeping the Random Forest model. We use XGBoost.

## Implementation Approach

1.  **Scaffold**: Copy the `MachineLearningPipeline` structure.
2.  **Adapt Step 1 (Data)**: Replace `create_dollar_bars` with `aggregate_to_volume_bars`.
3.  **Adapt Step 2 (Features)**: Replace `create_features` (FFD logic) with Palazzo's feature generation.
4.  **Adapt Step 3 (Labels)**: Replace TBM logic with `create_labels` and `compute_class_weight`. Derive `t1` series for Purged CV.
5.  **Adapt Run Loop**: Replace RF logic with XGBoost. Remove PCA. Ensure `PurgedKFold` receives the correct `t1`.

## Phase 1: Create `PalazzoXGBoostPipeline`

### Overview
Create the new pipeline file and implement the class structure, integrating the Palazzo logic into the defined steps.

### Changes Required:

#### 1. Create `src/modeling/xgboost_pipeline_palazzo.py`
**File**: `src/modeling/xgboost_pipeline_palazzo.py`
**Content**:
- Copy necessary imports from both source files.
- helper functions from `xgboost_price_reversal_palazzo.py` (e.g., `aggregate_to_volume_bars`, `create_labels`, `create_features`, and all indicator functions like `rsi_indicator`, `macd`, etc.). *Note: We might need to copy the indicator functions if they aren't importable, or import them if they are in `src/data_analysis/indicators.py`. Palazzo has many custom implementations inline.*
- **Action**: Copy the inline indicator functions from `xgboost_price_reversal_palazzo.py` into this new file (or a new utils file) to ensure self-containment, as they differ slightly from the shared library.

**Class Structure**:
```python
class PalazzoXGBoostPipeline:
    def __init__(self, config):
        self.config = config

    def step_1_data_structuring(self, raw_tick_data):
        # Calls aggregate_to_volume_bars
        pass

    def step_2_feature_engineering(self, bars):
        # Calls create_features (Palazzo version)
        pass

    def step_3_labeling_and_weighting(self, df):
        # Calls create_labels
        # Computes t1 (next bar timestamp)
        # Computes sample_weights (balanced class weights)
        pass

    def run(self, raw_tick_data, model):
        # Orchestrates the steps
        # Uses PurgedKFold with t1
        # Fits XGBoost
        pass
```

#### 2. Main Execution Block
- Add a `main()` function similar to `statistical_methods.py` but configuring XGBoost and the Palazzo parameters (volume threshold, etc.).

### Success Criteria:

#### Automated Verification:
- [x] File exists: `ls src/modeling/xgboost_pipeline_palazzo.py`
- [x] Syntax check: `python3 -m py_compile src/modeling/xgboost_pipeline_palazzo.py`
- [x] Imports resolve (dependencies like `xgboost`, `pandas`, `cupy` are installed).

#### Manual Verification:
- [ ] Run the script: `python3 src/modeling/xgboost_pipeline_palazzo.py`
- [ ] Output shows "Aggregation complete", "Labeling complete", and "Backtest Classification Report".
- [ ] F1 score is generated using Purged Cross-Validation.

---

## Testing Strategy

### Manual Testing Steps:
1.  Run the pipeline with a small subset of data (or the existing data path).
2.  Verify that Volume Bars are created (check output logs for bar count).
3.  Verify that features include `feature_RSI_pct`, etc.
4.  Verify that `PurgedKFold` splits are occurring (check logs for "Fold X...").

## References
- `thoughts/shared/research/2026-02-12-ml-cycle-modeling-comparison.md`
- `src/modeling/statistical_methods.py`
- `src/modeling/xgboost_price_reversal_palazzo.py`
