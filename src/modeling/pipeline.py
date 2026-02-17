from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.base import clone
from src.modeling import PurgedKFold
from src.data_analysis.data_analysis import timer

class AbstractMLPipeline(ABC):
    """
    Abstract Base Class for Machine Learning Pipelines.
    Consolidates the structure: Data Structuring -> Feature Engineering -> Labeling -> Purged CV.
    """

    def __init__(self, config):
        self.config = config
        self.problem_type = "classification" # Default to classification

    @abstractmethod
    def step_1_data_structuring(self, raw_tick_data):
        """Generate information-driven bars (e.g., Dollar Bars, Volume Bars)."""
        pass

    @abstractmethod
    def step_2_feature_engineering(self, bars):
        """Create features and ensure stationarity."""
        pass

    @abstractmethod
    def step_3_labeling_and_weighting(self, bars):
        """
        Define target labels and sample weights.
        Should return: (labels, sample_weights, t1)
        - labels: pd.Series or pd.DataFrame with target column.
        - sample_weights: pd.Series with same index as labels.
        - t1: pd.Series with event end times (for PurgedKFold).
        """
        pass

    def log_config(self, logger):
        """
        Log pipeline configuration and metadata to MLflow.
        """
        logger.log_params(self.config, prefix="pipeline_config")
        logger.log_params({"pipeline_class": self.__class__.__name__})

    def log_results(self, logger, model, X_test=None, y_test=None):
        """
        Hook for logging custom results after model training.
        Can be overridden by subclasses to log specific model artifacts (e.g. AutoGluon leaderboard).
        """
        pass

    @timer
    def run_cv(self, raw_tick_data, model):
        """
        Executes the full pipeline with Purged Cross-Validation.
        Returns:
            trained_model: Model fitted on full dataset.
            scores: List of F1 scores from each fold.
            X_final: Transformed feature matrix of the full dataset.
            y_final: Target vector of the full dataset.
            sw_final: Sample weights of the full dataset.
            t1_final: Event end times of the full dataset.
        """
        # Part I: Data Analysis
        bars = self.step_1_data_structuring(raw_tick_data)
        features = self.step_2_feature_engineering(bars)
        labels, sample_weights, t1 = self.step_3_labeling_and_weighting(bars)

        # 2. Alignment
        # We ensure all components share the same indices
        common_idx = features.index.intersection(labels.index).intersection(sample_weights.index).intersection(t1.index)
        X_raw = features.loc[common_idx]
        y = labels.loc[common_idx]
        sw = sample_weights.loc[common_idx]
        t1_series = t1.loc[common_idx]

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        # Part II: Modeling with Purged Cross-Validation
        cv = PurgedKFold(
            n_splits=self.config["n_splits"], t1=t1_series, pct_embargo=self.config["pct_embargo"]
        )

        scores = []
        
        print(f"Starting Purged Cross-Validation ({self.config['n_splits']} folds)...")
        for i, (train_idx, test_idx) in enumerate(cv.split(X_raw, y)):
            # 1. Split
            X_train_raw, X_test_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            sw_train = sw.iloc[train_idx]

            # 2. Fit Scaler on TRAIN only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            # 3. Fit PCA on TRAIN only (Optional)
            if self.config.get("use_pca", False):
                pca_fold = PCA(n_components=self.config.get("pca_components", 0.95), random_state=42)
                X_train_transformed = pca_fold.fit_transform(X_train_scaled)
                X_test_transformed = pca_fold.transform(X_test_scaled)
            else:
                X_train_transformed = X_train_scaled
                X_test_transformed = X_test_scaled

            # 4. Fit Model
            fold_model = clone(model)
            fold_model.fit(X_train_transformed, y_train, sample_weight=sw_train.values)
            
            # 5. Predict and Score
            y_pred = fold_model.predict(X_test_transformed)
            scores.append(f1_score(y_test, y_pred, average="weighted"))
            print(f"Fold {i+1} F1: {scores[-1]:.4f}")

        # --- Final Fit on Full Dataset ---
        scaler_final = StandardScaler()
        X_scaled_final = scaler_final.fit_transform(X_raw)
        
        if self.config.get("use_pca", False):
            pca_final = PCA(n_components=self.config.get("pca_components", 0.95), random_state=42)
            X_final = pd.DataFrame(
                pca_final.fit_transform(X_scaled_final),
                index=X_raw.index,
                columns=[f"PC{i+1}" for i in range(pca_final.n_components_)] if hasattr(pca_final, 'n_components_') else None
            )
        else:
            X_final = pd.DataFrame(X_scaled_final, index=X_raw.index, columns=X_raw.columns)
            pca_final = None

        trained_model = clone(model)
        trained_model.fit(X_final, y, sample_weight=sw.values)

        return trained_model, scores, X_final, y, sw, t1_series, pca_final
