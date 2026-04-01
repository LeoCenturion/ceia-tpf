# Refactor Machine Learning Pipelines into Abstract Base Class

## Overview

We will refactor `src/modeling/xgboost_pipeline_palazzo.py` and `src/modeling/statistical_methods.py` to inherit from a new abstract base class, `AbstractMLPipeline`, located in `src/modeling/pipeline.py`. This will promote code reuse, standardize the pipeline structure (Data Structuring -> Feature Engineering -> Labeling -> Modeling), and simplify future pipeline additions.

## Current State Analysis

- **`statistical_methods.py`**: Contains `MachineLearningPipeline`. Uses Dollar Bars, FFD stationarity, Triple-Barrier Method, and Random Forest.
- **`xgboost_pipeline_palazzo.py`**: Contains `PalazzoXGBoostPipeline`. Uses Volume Bars, Percentage Change stationarity, Dynamic Reversal Labeling, and XGBoost.
- **Commonalities**:
    - Both follow a 3-step preparation process: Data Structuring, Feature Engineering, Labeling/Weighting.
    - Both use `PurgedKFold` cross-validation.
    - Both use `StandardScaler` and optionally `PCA`.
    - Both use `timer` decorator.
- **Differences**:
    - Specific implementations of the 3 steps.
    - Model types (RF vs XGB).
    - `xgboost_pipeline_palazzo.py` lacks the extensive feature importance analysis present in `statistical_methods.py`.

## Desired End State

- **`src/modeling/pipeline.py`**: A new file containing `AbstractMLPipeline`.
    - Defines abstract methods: `step_1_data_structuring`, `step_2_feature_engineering`, `step_3_labeling_and_weighting`.
    - Implements concrete method: `run_cv` (consolidating the `run` logic) and `fit_predict`.
- **`src/modeling/statistical_methods.py`**: Refactored to inherit from `AbstractMLPipeline`.
- **`src/modeling/xgboost_pipeline_palazzo.py`**: Refactored to inherit from `AbstractMLPipeline`.

## Implementation Approach

1.  **Create `AbstractMLPipeline`**: Define the skeleton and move the common `run`/`run_cv` logic here. We'll need to handle model differences (e.g., `sample_weight` passing).
2.  **Refactor `statistical_methods.py`**: Make it inherit from the base class and implement the specific steps.
3.  **Refactor `xgboost_pipeline_palazzo.py`**: Make it inherit from the base class.

## Phase 1: Create Abstract Base Class

### Overview
Create `src/modeling/pipeline.py` with `AbstractMLPipeline`.

### Changes Required:

#### 1. `src/modeling/pipeline.py`
**File**: `src/modeling/pipeline.py`
**Content**:
```python
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from src.modeling import PurgedKFold
import pandas as pd
import numpy as np
from src.data_analysis.data_analysis import timer

class AbstractMLPipeline(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def step_1_data_structuring(self, raw_tick_data):
        pass

    @abstractmethod
    def step_2_feature_engineering(self, bars):
        pass

    @abstractmethod
    def step_3_labeling_and_weighting(self, bars):
        """Should return labels (y), sample_weights, and t1 (event end times)."""
        pass

    @timer
    def run_cv(self, raw_tick_data, model):
        # ... logic extracted from existing run methods ...
        # 1. Execute steps
        bars = self.step_1_data_structuring(raw_tick_data)
        features = self.step_2_feature_engineering(bars)
        labels, weights, t1 = self.step_3_labeling_and_weighting(bars)

        # 2. Alignment
        common_idx = features.index.intersection(labels.index).intersection(weights.index).intersection(t1.index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        sw = weights.loc[common_idx]
        t1 = t1.loc[common_idx]
        
        # Handle y being a DataFrame or Series
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        # 3. CV Loop
        cv = PurgedKFold(n_splits=self.config["n_splits"], t1=t1, pct_embargo=self.config["pct_embargo"])
        scores = []
        
        for train_idx, test_idx in cv.split(X, y):
            # ... split, scale, pca, fit, predict ...
            pass
            
        return scores, X, y, sw, t1
```

### Success Criteria:
- [ ] `src/modeling/pipeline.py` exists.
- [ ] Abstract methods are defined.
- [ ] Common imports are present.

## Phase 2: Refactor Statistical Methods Pipeline

### Overview
Update `src/modeling/statistical_methods.py` to use the new base class.

### Changes Required:
- Import `AbstractMLPipeline`.
- Inherit `MachineLearningPipeline` from `AbstractMLPipeline`.
- Remove `run` method (use base class `run_cv` or override if significant diffs exist). *Note: statistical_methods.py returns many artifacts (trained_model, X_final, etc.) for feature importance. We might need `run_cv` to return these or have a separate method.*

### Success Criteria:
- [ ] `poetry run python -m src.modeling.statistical_methods` runs successfully.

## Phase 3: Refactor Palazzo XGBoost Pipeline

### Overview
Update `src/modeling/xgboost_pipeline_palazzo.py` to use the new base class.

### Changes Required:
- Import `AbstractMLPipeline`.
- Inherit `PalazzoXGBoostPipeline` from `AbstractMLPipeline`.
- Adapt to the base class interface (e.g., return values of step 3).

### Success Criteria:
- [ ] `poetry run python -m src.modeling.xgboost_pipeline_palazzo` runs successfully.

## Verification Plan

### Automated Verification:
- [ ] Run both scripts:
    - `poetry run python -m src.modeling.statistical_methods`
    - `poetry run python -m src.modeling.xgboost_pipeline_palazzo`

### Manual Verification:
- [ ] Check output logs for expected F1 scores and execution steps.
