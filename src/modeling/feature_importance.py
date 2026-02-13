import time
from functools import wraps
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, f1_score
from sklearn.base import clone
from scipy.stats import weightedtau

def timer(func):
    """Decorator that prints the execution time of the function it decorates."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"\n>>> Function '{func.__name__}' executed in {duration:.4f} seconds")
        return result

    return wrapper

@timer
def feature_importance_mdi(model, X, y):
    """
    Mean Decrease Impurity (MDI) feature importance.
    Trains a new RandomForestClassifier with max_features=1 to avoid masking effects.
    """
    # Use a new model with the specified max_features
    # Note: We assume model is an ensemble like RF. If it's XGB, this might need adjustment,
    # but the original code explicitly created a RandomForestClassifier.
    # We will stick to the original implementation which forces RF for MDI calculation.
    mdi_model = RandomForestClassifier(
        n_estimators=model.n_estimators,
        max_features=1,  # Key change for López de Prado's method
        random_state=model.random_state,
        n_jobs=model.n_jobs,
    )
    mdi_model.fit(X, y)

    importances = pd.Series(mdi_model.feature_importances_, index=X.columns, name="MDI")
    return importances.sort_values(ascending=False)


def _get_mda_score_for_feature(
    model, X_test, y_test, col, base_score, scorer, sample_weights, labels=None
):
    """Helper for MDA: Permutes a single column and returns the score difference."""
    X_test_permuted = X_test.copy()
    np.random.shuffle(X_test_permuted[col].values)

    y_pred_permuted = (
        model.predict_proba(X_test_permuted)
        if scorer == log_loss
        else model.predict(X_test_permuted)
    )

    if scorer == log_loss and labels is not None:
        permuted_score = scorer(
            y_test, y_pred_permuted, sample_weight=sample_weights, labels=labels
        )
    else:
        permuted_score = scorer(y_test, y_pred_permuted, sample_weight=sample_weights)

    return base_score - permuted_score


@timer
def feature_importance_mda(model, X, y, cv, sample_weights, t1, scoring="neg_log_loss"):
    """
    Mean Decrease Accuracy (MDA) feature importance using PurgedKFold.
    The permutation of features is parallelized.
    """
    if scoring == "neg_log_loss":
        scorer = log_loss
    else:
        # Assuming f1_score with "weighted" average
        scorer = lambda y_true, y_pred, sample_weight, labels=None: f1_score(
            y_true, y_pred, average="weighted", sample_weight=sample_weight
        )

    fold_importances = []
    for train_idx, test_idx in cv.split(X, y, groups=t1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        sample_weight_train = sample_weights.iloc[train_idx]
        sample_weight_test = sample_weights.iloc[test_idx]

        model.fit(X_train, y_train, sample_weight=sample_weight_train.values)

        # Capture classes from the trained model
        labels = model.classes_

        y_pred = (
            model.predict_proba(X_test) if scorer == log_loss else model.predict(X_test)
        )

        if scorer == log_loss:
            base_score = scorer(
                y_test, y_pred, sample_weight=sample_weight_test.values, labels=labels
            )
        else:
            base_score = scorer(y_test, y_pred, sample_weight=sample_weight_test.values)

        # Sequential evaluation of each feature to avoid multiprocessing issues
        feature_scores = []
        for col in X.columns:
            # _get_mda_score_for_feature returns (base_score - permuted_score)
            diff = _get_mda_score_for_feature(
                model,
                X_test,
                y_test,
                col,
                base_score,
                scorer,
                sample_weight_test.values,
                labels=labels if scorer == log_loss else None,
            )
            
            # If Loss (log_loss): diff = Loss_orig - Loss_perm. 
            # We want Loss_perm - Loss_orig, so we negate it.
            # If Accuracy (f1): diff = Acc_orig - Acc_perm.
            # We want Acc_orig - Acc_perm, so we keep it.
            if scorer == log_loss:
                importance = -diff
            else:
                importance = diff
                
            feature_scores.append(importance)

        fold_importances.append(pd.Series(feature_scores, index=X.columns))

    # Average importance across all folds
    importances = pd.concat(fold_importances, axis=1).mean(axis=1)

    # Add the full model score for reference (keeping the raw base_score from the last fold)
    importances.loc["full_model"] = base_score

    # Sort from most predictive to least predictive
    return importances.sort_values(ascending=False)


def _run_sfi_for_feature(model, X, y, col, cv, sample_weights, t1):
    """Helper for SFI: Runs a full CV for a single feature."""
    scores = []
    for train_idx, test_idx in cv.split(X, y, groups=t1):
        X_train, X_test = X[[col]].iloc[train_idx], X[[col]].iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        sample_weight_train = sample_weights.iloc[train_idx]

        model.fit(X_train, y_train, sample_weight=sample_weight_train.values)
        y_pred = model.predict(X_test)

        score = f1_score(y_test, y_pred, average="weighted")
        scores.append(score)
    return np.mean(scores)


@timer
def feature_importance_sfi(model, X, y, cv, sample_weights, t1, scoring="f1_weighted"):
    """
    Single Feature Importance (SFI) using PurgedKFold.
    """
    # Use a clean, unfitted model for each feature to avoid pickling overhead and state leakage
    base_model = clone(model)

    # Sequential evaluation of each feature
    scores = []
    for col in X.columns:
        score = _run_sfi_for_feature(base_model, X, y, col, cv, sample_weights, t1)
        scores.append(score)

    importances = pd.Series(scores, index=X.columns)
    return importances.sort_values(ascending=False)


@timer
def feature_importance_orthogonal(model, X_ortho, y, sample_weights, pca):
    """
    Feature importance on orthogonal features (PCA components).
    """
    model.fit(X_ortho, y, sample_weight=sample_weights.values)
    importances_ortho = pd.Series(
        model.feature_importances_, index=X_ortho.columns, name="Orthogonal Importance"
    )

    # Combine with PCA explained variance
    explained_variance = pd.Series(
        pca.explained_variance_ratio_, index=X_ortho.columns, name="Explained Variance"
    )

    results = pd.concat([importances_ortho, explained_variance], axis=1)
    results = results.sort_values(by="Orthogonal Importance", ascending=False)

    return results


def weighted_kendalls_tau(series1, series2):
    """
    Computes the weighted Kendall's tau rank correlation between two series.
    """
    # Align the two series by their index
    aligned_s1, aligned_s2 = series1.align(series2, join="inner")

    if aligned_s1.empty or aligned_s2.empty:
        return np.nan, np.nan

    correlation, p_value = weightedtau(aligned_s1, aligned_s2)

    return correlation, p_value
