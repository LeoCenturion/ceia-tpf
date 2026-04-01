# AutoGluon Integration Plan

## Overview
Integrate AutoGluon's `TabularPredictor` into the `Palazzo` financial modeling pipeline to enable AutoML capabilities for both direct price reversal prediction and meta-labeling (filtering primary model predictions).

## Current State Analysis
- **Existing Pipeline**: `PalazzoXGBoostPipeline` (in `src/modeling/xgboost_pipeline_palazzo.py`) handles data structuring (volume bars), feature engineering (technical indicators), and labeling (Triple Barrier).
- **Coupling**: The pipeline uses `PurgedKFold` and expects an estimator with a standard `fit(X, y, sample_weight)` / `predict(X)` interface.
- **Gap**: AutoGluon uses a `fit(train_data, label=...)` interface and handles internal validation, which conflicts slightly with the manual CV loop if not adapted correctly.

## Desired End State
1.  **Standalone AutoGluon Pipeline**: A new pipeline class that uses AutoGluon instead of XGBoost to predict price reversals, reusing the exact same features and labeling logic.
2.  **Meta-Labeling AutoGluon Pipeline**: A new pipeline class that:
    - Splits data into `Train_CV` and `Test_Holdout`.
    - Generates Out-Of-Fold (OOF) predictions from a primary XGBoost model on `Train_CV`.
    - Trains an AutoGluon meta-model on these OOF predictions (plus original features) to predict "Correctness" (Meta-Label).
    - Evaluates the combined system (Primary + Meta-Filter) on `Test_Holdout`.

## Implementation Approach
- **Adapter Pattern**: Create an `AutoGluonAdapter` class that wraps `TabularPredictor` to look like an `sklearn` classifier.
- **Inheritance**: Subclass `PalazzoXGBoostPipeline` to reuse feature engineering.
- **Stacking Workflow**: Implement a specific method to gather OOF predictions from the primary model to form the meta-training set, ensuring no leakage.

## Phase 1: AutoGluon Adapter & Standalone Pipeline

### Overview
Create the adapter and the standalone pipeline to verify AutoGluon works within the `Palazzo` architecture.

### Changes Required:

#### 1. AutoGluon Adapter
**File**: `src/modeling/autogluon_adapter.py` (New File)
**Instruction**: Create a class `AutoGluonAdapter` that inherits from `sklearn.base.BaseEstimator` and `ClassifierMixin`.
- **`__init__`**: Accept `time_limit`, `presets`, `eval_metric`, etc.
- **`fit(X, y, sample_weight=None)`**: 
    - Combine `X` and `y` into a temporary DataFrame.
    - Call `TabularPredictor.fit()`.
    - Handle `sample_weight` if AutoGluon supports it.
- **`predict(X)`**: Call `TabularPredictor.predict()`.
- **`predict_proba(X)`**: Call `TabularPredictor.predict_proba()`.

#### 2. Standalone Pipeline
**File**: `src/modeling/autogluon_pipeline_palazzo.py` (New File)
**Instruction**: Create `PalazzoAutoGluonPipeline` inheriting from `PalazzoXGBoostPipeline`.
- Import `AutoGluonAdapter`.
- Override `main()`:
    - Instantiate `PalazzoAutoGluonPipeline`.
    - Instantiate `AutoGluonAdapter` with desired config (e.g., `presets='medium_quality'`).
    - Call `pipeline.run_cv(data, adapter_model)`.

### Success Criteria:
#### Automated Verification:
- [x] Pipeline runs without errors: `python src/modeling/autogluon_pipeline_palazzo.py`
- [x] Adapter creates a valid AutoGluon model directory.

#### Manual Verification:
- [x] Check if scores are comparable/sane compared to XGBoost baseline.

---

## Phase 2: Meta-Labeling Pipeline

### Overview
Implement the meta-labeling logic using OOF predictions for training and a held-out set for testing.

### Changes Required:

#### 1. Meta-Labeling Pipeline class
**File**: `src/modeling/autogluon_metalabeling_palazzo.py` (New File)
**Instruction**: Create `PalazzoMetaLabelingPipeline` inheriting from `PalazzoXGBoostPipeline`.

- **Method**: `generate_oof_predictions(self, raw_tick_data, model)`
    - **Logic**:
        - Perform Step 1 (Structuring), Step 2 (Features), Step 3 (Labeling) on `raw_tick_data`.
        - Initialize `PurgedKFold`.
        - **Loop over Folds**:
            - Fit `model` on `Train_Fold`.
            - Predict on `Validation_Fold`.
            - Store predictions, probabilities, and true labels aligned with the `Validation_Fold` index.
    - **Return**: DataFrame containing `[true_label, primary_pred, primary_prob]` for all validation samples.

- **Method**: `run_metalabeling_experiment(self, raw_tick_data)`
    - **Logic**:
        1.  **Split Data**: `Train_CV_Data` (e.g., first 80%) and `Test_Holdout_Data` (last 20%).
        2.  **Generate OOF**: Call `generate_oof_predictions(Train_CV_Data, xgb_model)`.
        3.  **Create Meta-Labels**: 
            - `meta_label = 1` if `primary_pred == true_label` else `0`.
            - **Feature Set**: Use Original Features of `Train_CV_Data` + `primary_prob` column.
        4.  **Train Meta-Model**: Fit `AutoGluonAdapter` on this Meta-Dataset.
        5.  **Train Primary Final**: Fit `xgb_model` on *all* `Train_CV_Data`.
        6.  **Evaluate on Holdout**:
            - Process `Test_Holdout_Data` to get features.
            - `primary_preds` = `primary_final.predict(Test_Features)`.
            - `meta_preds` = `autogluon.predict(Test_Features + primary_probs)`.
            - **Final Decision**: Trade if `primary_preds` indicates trade AND `meta_preds == 1`.
            - Report classification metrics.

- **Main Execution**:
    - Load data.
    - Run `run_metalabeling_experiment`.

### Success Criteria:
#### Automated Verification:
- [x] Meta-labeling script runs: `python src/modeling/autogluon_metalabeling_palazzo.py`
- [x] OOF predictions are generated for the training set.

#### Manual Verification:
- [x] Verify that the meta-model improves precision on the held-out test set compared to the standalone primary model.
- [x] Confirm no leakage: The meta-model is trained only on OOF predictions from the CV set.

---

## Testing Strategy

### Unit Tests:
- N/A

### Manual Testing Steps:
1.  Run `src/modeling/autogluon_pipeline_palazzo.py` (Phase 1).
2.  Run `src/modeling/autogluon_metalabeling_palazzo.py` (Phase 2).
3.  Compare the "Final Classification Report" of the meta-labeling script against the baseline XGBoost report.

## Performance Considerations
- **Memory**: Accumulating OOF predictions is memory efficient.
- **Time**: Running CV + Meta Training + Final Training will take longer.

## Migration Notes
- N/A
