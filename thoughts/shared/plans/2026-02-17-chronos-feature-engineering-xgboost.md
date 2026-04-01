# Chronos Feature Engineering for XGBoost Implementation Plan

## Overview
This plan outlines the creation of a new machine learning pipeline that leverages a pre-trained Chronos model for advanced time-series feature engineering. The resulting Chronos embeddings will be combined with existing tabular features and fed into an XGBoost model for classification tasks, integrating seamlessly with the existing MLflow tracking framework.

## Current State Analysis
- Existing pipelines (`PalazzoXGBoostPipeline`, `PalazzoAutoGluonPipeline`, `PalazzoMetaLabelingPipeline`, `PalazzoChronosPipeline`) provide a modular structure for data structuring, feature engineering, labeling, and cross-validation.
- `pipeline_runner.py` handles the overall MLflow experiment management and execution flow.
- The `AbstractMLPipeline` class defines a common interface and hooks for configuration and results logging.
- `chronos_pipeline_palazzo.py` demonstrates the use of Chronos for forecasting, including custom feature filtering.

## Desired End State
A new, independent pipeline (`ChronosFeaturePipeline`) that:
- Utilizes Chronos for generating rich latent representations (embeddings) from time-series data.
- Combines these Chronos embeddings with existing tabular features.
- Trains an XGBoost classifier on the combined feature set.
- Integrates with `run_pipeline` for MLflow experiment tracking, including logging of parameters, metrics, and classification reports.

### Key Discoveries:
- The `PalazzoXGBoostPipeline` provides a solid base for data structuring (`step_1_data_structuring`) and labeling (`step_3_labeling_and_weighting`).
- The `run_pipeline` function in `pipeline_runner.py` is capable of orchestrating different pipeline implementations, including passing `model_cls` and `model_params` for model instantiation.
- The `mlflow_utils.py` provides helpers for logging various data types to MLflow.

## What We're NOT Doing
- Re-training the Chronos model from scratch. We will use a pre-trained Chronos-T5 model.
- Modifying the core `run_pipeline` function for this specific pipeline beyond necessary argument passing.
- Implementing new data structuring or labeling logic, as this will be reused from `PalazzoXGBoostPipeline`.

## Implementation Approach
High-level strategy involves creating a new pipeline class that overrides the feature engineering step to incorporate Chronos-based embeddings, then using `run_pipeline` to manage the experiment with XGBoost.

## Phase 1: Create New Pipeline File and Class Structure

### Overview
Create the basic file structure and define the `ChronosFeaturePipeline` class inheriting from `PalazzoXGBoostPipeline`.

### Changes Required:

#### 1. New File: `src/modeling/chronos_feature_pipeline.py`
**File**: `src/modeling/chronos_feature_pipeline.py`
**Changes**: Create the file with the basic class definition and necessary imports.

```python
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Transformers and AutoGluon for Chronos integration
from transformers import AutoModelForPrediction, AutoTokenizer, T5EncoderModel
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame # Might not need TimeSeriesPredictor if only using encoder

# Make the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_analysis.data_analysis import fetch_historical_data, timer
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
from src.modeling.pipeline_runner import run_pipeline
from src.modeling.autogluon_adapter import AutoGluonAdapter # For consistency if needed, but Chronos itself here
from src.modeling import PurgedKFold
from src.constants import VOLUME_COL, CLOSE_COL

class ChronosFeaturePipeline(PalazzoXGBoostPipeline):
    """
    Pipeline that uses Chronos for feature engineering, feeding embeddings into XGBoost.
    """
    def __init__(self, config):
        super().__init__(config)
        self.problem_type = "classification"
        self.chronos_model = None
        self.chronos_tokenizer = None
        self.chronos_encoder = None

    @timer
    def step_2_feature_engineering(self, bars):
        # This method will be implemented in subsequent phases
        pass

    # The run and log_results methods will likely be inherited or slightly adapted from parent/pipeline_runner
    # but for now, we'll focus on feature engineering.

def main():
    # Placeholder for main function setup
    pass

if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [ ] File `src/modeling/chronos_feature_pipeline.py` exists.
- [ ] The file contains the `ChronosFeaturePipeline` class inheriting from `PalazzoXGBoostPipeline`.
- [ ] The basic imports are present.
- [ ] No immediate syntax errors on file creation.

#### Manual Verification:
- [ ] The file structure is clean and adheres to project conventions.

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to the next phase.

---

## Phase 2: Implement Chronos Preprocessing and Embedding Extraction

### Overview
Implement the core logic within `step_2_feature_engineering` to load Chronos components, preprocess time-series data, and extract embeddings.

### Changes Required:

#### 1. Modify `src/modeling/chronos_feature_pipeline.py`
**File**: `src/modeling/chronos_feature_pipeline.py`
**Changes**: Implement the `step_2_feature_engineering` method.

```python
    @timer
    def step_2_feature_engineering(self, bars):
        logging.debug("Step 2: Generating Chronos features and combining with tabular...")

        # 1. Generate standard tabular features using parent logic
        tabular_features = super().step_2_feature_engineering(bars) # This already calls dropna()

        # Ensure we have a clean index after dropping NaNs from parent feature engineering
        common_index = tabular_features.index.intersection(bars.index)
        bars_aligned = bars.loc[common_index]

        # 2. Load Chronos components (if not already loaded)
        if self.chronos_model is None:
            # Use a smaller model for faster iteration, can be configured later
            chronos_model_name = self.config.get("chronos_model_name", "amazon/chronos-t5-tiny")
            self.chronos_tokenizer = AutoTokenizer.from_pretrained(chronos_model_name)
            self.chronos_encoder = T5EncoderModel.from_pretrained(chronos_model_name)
            # Move to GPU if available
            if torch.cuda.is_available():
                self.chronos_encoder.to("cuda")

        # 3. Chronos Preprocessing: Scaling, Quantization, and Tokenization
        # We need a time-series input for Chronos. Let's use 'close_price' from bars_aligned
        # For feature extraction, we process windows of the time series.
        # Let's define a window_size and a stride.
        window_size = self.config.get("chronos_window_size", 128) # Example window size
        stride = self.config.get("chronos_stride", 1) # Example stride

        chronos_embeddings = []
        for i in range(0, len(bars_aligned) - window_size + 1, stride):
            window = bars_aligned.iloc[i : i + window_size]
            time_series_data = window["close_price"].values.astype(np.float32)

            # Apply Chronos-specific scaling (mean scaling)
            mean_abs_value = np.mean(np.abs(time_series_data))
            if mean_abs_value == 0: # Avoid division by zero
                scaled_data = time_series_data
            else:
                scaled_data = time_series_data / mean_abs_value
            
            # Quantization and tokenization
            # The tokenizer handles quantization implicitly when processing numerical inputs
            inputs = self.chronos_tokenizer(scaled_data, return_tensors="pt", padding=True, truncation=True)
            
            # Move inputs to GPU if encoder is on GPU
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Extract embeddings from the encoder
            with torch.no_grad(): # Disable gradient calculation for inference
                encoder_outputs = self.chronos_encoder(**inputs)

            # Pooling: Use the embedding of the last token (or mean pooling)
            # For simplicity, let's use mean pooling across the sequence dimension
            # encoder_outputs.last_hidden_state has shape (batch_size, sequence_length, hidden_size)
            embedding = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            chronos_embeddings.append(embedding)

        # Convert list of embeddings to DataFrame
        if chronos_embeddings:
            chronos_features_df = pd.DataFrame(np.array(chronos_embeddings),
                                             index=bars_aligned.index[window_size - 1::stride],
                                             columns=[f"chronos_embed_{j}" for j in range(embedding.shape[-1])])
        else:
            chronos_features_df = pd.DataFrame(index=pd.Index([]))

        # Align chronos_features_df to tabular_features index before concatenation
        final_common_index = tabular_features.index.intersection(chronos_features_df.index)
        aligned_tabular_features = tabular_features.loc[final_common_index]
        aligned_chronos_features = chronos_features_df.loc[final_common_index]

        # 4. Combine Chronos embeddings with tabular features
        combined_features = pd.concat([aligned_tabular_features, aligned_chronos_features], axis=1)
        
        return combined_features.dropna() # Ensure final features are clean
```

### Success Criteria:

#### Automated Verification:
- [ ] `src/modeling/chronos_feature_pipeline.py` can be imported without syntax errors.
- [ ] The `step_2_feature_engineering` method correctly loads Chronos components.
- [ ] The `step_2_feature_engineering` method produces a DataFrame with combined tabular and Chronos features.
- [ ] Unit tests for `step_2_feature_engineering` are added (to be defined later).

#### Manual Verification:
- [ ] The generated features appear reasonable (e.g., embedding dimensions are as expected).
- [ ] No unexpected warnings or errors during feature generation.

---

## Phase 3: Integrate with `run_pipeline` and `main` Function

### Overview
Finalize the `main` function to properly set up the pipeline, define model parameters, and execute the `run_pipeline` function for MLflow tracking.

### Changes Required:

#### 1. Modify `src/modeling/chronos_feature_pipeline.py`
**File**: `src/modeling/chronos_feature_pipeline.py`
**Changes**: Implement the `main` function.

```python
import torch # Add torch import for GPU checks
# ... (existing imports)

def main():
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
    )
    raw_data.rename(columns={VOLUME_COL: "volume", CLOSE_COL: "close"}, inplace=True)

    # Configuration for the ChronosFeaturePipeline
    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_splits": 3,
        "pct_embargo": 0.01,
        "use_pca": False, # PCA might be redundant with Chronos embeddings, can be experimented with
        "chronos_model_name": "amazon/chronos-t5-tiny",
        "chronos_window_size": 128,
        "chronos_stride": 1,
    }

    # Primary Model (XGBoost) parameters
    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
    }

    pipeline = ChronosFeaturePipeline(pipeline_config)

    run_pipeline(
        pipeline=pipeline,
        model_cls=xgb.XGBClassifier, # The final model to train on Chronos features
        raw_data=raw_data,
        model_params=model_params,
        experiment_name="Chronos_Feature_XGBoost_Pipeline",
        data_path=data_path,
    )

if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [ ] The `main` function executes without runtime errors.
- [ ] An MLflow run is created with the specified `experiment_name`.
- [ ] Pipeline configuration and model parameters are logged to MLflow.
- [ ] Classification metrics and reports are logged to MLflow.

#### Manual Verification:
- [ ] The MLflow UI shows a complete run for "Chronos_Feature_XGBoost_Pipeline" with expected parameters, metrics, and artifacts.
- [ ] The performance metrics are reasonable.

---

## Testing Strategy

### Unit Tests:
- Test `step_1_data_structuring` with various raw tick data inputs.
- Test `step_2_feature_engineering` with mock `bars` data to verify Chronos embedding generation and feature concatenation.
- Test `step_3_labeling_and_weighting` with mock `bars` data to verify label and weight generation.

### Integration Tests:
- Run the full pipeline with a smaller dataset to ensure end-to-end functionality and MLflow logging.

### Manual Testing Steps:
1. Execute the `chronos_feature_pipeline.py` script.
2. Start `mlflow ui` and observe the new run. Verify all parameters, metrics, and artifacts (especially the classification report JSONs) are present and correct.
3. Check console output for expected messages and no errors.

## Performance Considerations
- Chronos embedding generation can be computationally intensive, especially with large `window_size` or frequent time series. Consider optimizing the windowing and batching if performance is an issue.
- GPU usage for Chronos encoder should be enabled for performance.

## Migration Notes
- No specific migrations needed as this is a new pipeline.

## References
- Existing pipelines: `src/modeling/xgboost_pipeline_palazzo.py`, `src/modeling/chronos_pipeline_palazzo.py`
- MLflow utility: `src/modeling/mlflow_utils.py`
