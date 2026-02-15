import pandas as pd
import numpy as np
import shutil
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from autogluon.tabular import TabularPredictor

class AutoGluonAdapter(BaseEstimator, ClassifierMixin):
    """
    Adapter to make AutoGluon's TabularPredictor compatible with sklearn's interface.
    This allows it to be used in pipelines that expect fit(X, y) and predict(X).
    """
    def __init__(self, label='label', problem_type=None, eval_metric=None, 
                 time_limit=60, presets='medium_quality', hyperparameters=None, path=None, verbosity=2):
        self.label = label
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.time_limit = time_limit
        self.presets = presets
        self.hyperparameters = hyperparameters
        self.path = path
        self.verbosity = verbosity
        self.predictor = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the AutoGluon model.
        X: pd.DataFrame or np.array
        y: pd.Series or np.array
        sample_weight: pd.Series or np.array (optional)
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            # If columns are integers (from numpy array), give them string names to avoid AutoGluon issues
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]
        
        # specific handling for 'y' if it's a dataframe
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        
        # Use values to avoid index alignment issues (e.g. if X has RangeIndex and y has DatetimeIndex)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            train_data[self.label] = y.values
        else:
            train_data[self.label] = y
        
        # Handle sample weights if provided
        sample_weight_col = None
        if sample_weight is not None:
            sample_weight_col = 'sample_weight'
            if isinstance(sample_weight, (pd.Series, pd.DataFrame)):
                train_data[sample_weight_col] = sample_weight.values
            else:
                train_data[sample_weight_col] = sample_weight

        # Initialize predictor
        # We start a fresh predictor for each fit to mimic sklearn behavior (no warm start by default)
        fit_path = self.path
        if fit_path is not None:
            # Add a unique suffix to the path for each fit to avoid conflicts in CV
            import uuid
            fit_path = os.path.join(fit_path, str(uuid.uuid4())[:8])

        self.predictor = TabularPredictor(
            label=self.label,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            path=fit_path,
            sample_weight=sample_weight_col,
            verbosity=self.verbosity
        )

        # Fit
        self.predictor.fit(
            train_data,
            time_limit=self.time_limit,
            presets=self.presets,
            hyperparameters=self.hyperparameters,
            ag_args_fit={'num_gpus': 1} if self._check_gpu() else None
        )
        
        self.classes_ = self.predictor.class_labels
        return self

    def predict(self, X):
        """
        Predict class labels for X.
        """
        if self.predictor is None:
            raise RuntimeError("The model has not been fitted yet.")
            
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]

        return self.predictor.predict(X).values

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        if self.predictor is None:
            raise RuntimeError("The model has not been fitted yet.")
            
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]

        # AutoGluon returns a DataFrame with columns for each class
        probs = self.predictor.predict_proba(X)
        
        # Ensure the order matches self.classes_
        if self.classes_ is not None:
            # Reorder columns to match classes_
            probs = probs[self.classes_]
            
        return probs.values

    def _check_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def leaderboard(self, data=None, silent=False):
        """
        Output the leaderboard of trained models.
        """
        if self.predictor is None:
            print("Predictor not fitted yet.")
            return None
        return self.predictor.leaderboard(data, silent=silent)
