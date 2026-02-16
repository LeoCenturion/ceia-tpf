import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

# Make the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling import PurgedKFold
from src.constants import VOLUME_COL, CLOSE_COL


class PalazzoMetaLabelingPipeline(PalazzoXGBoostPipeline):
    """
    Pipeline that implements Meta-Labeling using AutoGluon.
    It splits data into Train/Test, uses CV on Train to generate OOF predictions,
    trains a Meta-Model to predict correctness of Primary Model,
    and then evaluates on Test.
    """

    def generate_oof_predictions(self, X, y, t1, sw, model):
        """
        Generates Out-Of-Fold predictions for the given dataset using PurgedKFold.
        Returns a DataFrame with [true_label, primary_pred, primary_prob]
        """
        cv = PurgedKFold(
            n_splits=self.config["n_splits"],
            t1=t1,
            pct_embargo=self.config["pct_embargo"],
        )

        oof_preds = pd.Series(index=X.index, dtype=float)
        oof_probs = pd.Series(index=X.index, dtype=float)
        fold_f1_scores = []

        print(f"Generating OOF predictions with {self.config['n_splits']} folds...")

        for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Split
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            sw_train = sw.iloc[train_idx]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # PCA (if enabled)
            if self.config.get("use_pca", False):
                from sklearn.decomposition import PCA

                pca = PCA(
                    n_components=self.config.get("pca_components", 0.95),
                    random_state=42,
                )
                X_train_final = pca.fit_transform(X_train_scaled)
                X_val_final = pca.transform(X_val_scaled)
            else:
                X_train_final = X_train_scaled
                X_val_final = X_val_scaled

            # Fit Primary Model
            fold_model = clone(model)
            fold_model.fit(X_train_final, y_train, sample_weight=sw_train.values)

            # Predict
            val_pred = fold_model.predict(X_val_final)
            val_prob = fold_model.predict_proba(X_val_final)[:, 1]

            # Score
            fold_f1 = f1_score(y_val, val_pred, average="weighted")
            fold_f1_scores.append(fold_f1)
            print(f"Fold {i + 1} F1 Score: {fold_f1:.4f}")

            # Store OOF
            # X.iloc[val_idx].index gives the original indices for this fold
            fold_indices = X.iloc[val_idx].index
            oof_preds.loc[fold_indices] = val_pred
            oof_probs.loc[fold_indices] = val_prob

        print(f"Average OOF CV F1 Score: {np.mean(fold_f1_scores):.4f}")

        return pd.DataFrame(
            {"true_label": y, "primary_pred": oof_preds, "primary_prob": oof_probs},
            index=X.index,
        )

    def run_metalabeling_experiment(self, raw_data, primary_model, meta_model_config):
        """
        Runs the full meta-labeling experiment.
        """
        # 1. Preprocessing (Step 1, 2, 3) on FULL data
        bars = self.step_1_data_structuring(raw_data)
        features = self.step_2_feature_engineering(bars)
        labels, sample_weights, t1 = self.step_3_labeling_and_weighting(bars)

        # Alignment
        common_idx = (
            features.index.intersection(labels.index)
            .intersection(sample_weights.index)
            .intersection(t1.index)
        )
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        sw = sample_weights.loc[common_idx]
        t1 = t1.loc[common_idx]

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        # 2. Split into Train_CV (80%) and Test_Holdout (20%)
        # Time-series split (no shuffle)
        split_idx = int(len(X) * 0.8)

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        sw_train = sw.iloc[:split_idx]
        t1_train = t1.iloc[:split_idx]

        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        t1_test = t1.iloc[
            split_idx:
        ]  # Not strictly needed for test eval but good for consistency

        print(f"Data Split: Train={len(X_train)}, Test={len(X_test)}")

        # 3. Generate OOF Predictions on Train_CV
        oof_df = self.generate_oof_predictions(
            X_train, y_train, t1_train, sw_train, primary_model
        )

        # 4. Create Meta-Labels and Meta-Dataset
        # Meta Label: 1 if Primary Model was Correct, 0 otherwise
        meta_labels = (oof_df["primary_pred"] == oof_df["true_label"]).astype(int)

        # Meta Features: Original Features + Primary Probability
        # (Could also add Primary Pred, but Prob is richer)
        X_meta_train = X_train.copy()
        X_meta_train["primary_prob"] = oof_df["primary_prob"]
        X_meta_train["primary_pred"] = oof_df[
            "primary_pred"
        ]  # Optional: Include the hard prediction

        print(
            f"Meta-Label Distribution: {meta_labels.value_counts(normalize=True).to_dict()}"
        )

        # 5. Train Meta-Model (AutoGluon)
        print("Training Meta-Model (AutoGluon)...")
        meta_model = AutoGluonAdapter(**meta_model_config)
        # We don't use sample weights for meta-model for now (or could use same weights)
        meta_model.fit(X_meta_train, meta_labels)

        # 6. Train Final Primary Model on ALL Train_CV
        print("Training Final Primary Model on all Training Data...")

        # Scale/PCA on full train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        if self.config.get("use_pca", False):
            from sklearn.decomposition import PCA

            pca = PCA(
                n_components=self.config.get("pca_components", 0.95), random_state=42
            )
            X_train_final = pca.fit_transform(X_train_scaled)
        else:
            X_train_final = X_train_scaled
            pca = None

        primary_final = clone(primary_model)
        primary_final.fit(X_train_final, y_train, sample_weight=sw_train.values)

        # 7. Evaluate on Test_Holdout
        print("\n--- Evaluation on Holdout Test Set ---")

        # Prepare Test Features
        X_test_scaled = scaler.transform(X_test)
        if pca:
            X_test_final = pca.transform(X_test_scaled)
        else:
            X_test_final = X_test_scaled

        # Primary Predictions on Test
        primary_test_pred = primary_final.predict(X_test_final)
        primary_test_prob = primary_final.predict_proba(X_test_final)[:, 1]

        # Meta Features for Test
        X_meta_test = X_test.copy()
        X_meta_test["primary_prob"] = primary_test_prob
        X_meta_test["primary_pred"] = primary_test_pred

        # Meta Predictions on Test
        meta_test_pred = meta_model.predict(
            X_meta_test
        )  # 1 = "Primary is Correct", 0 = "Primary is Wrong"

        # --- Metrics ---

        # 1. Baseline: Primary Model Alone
        print("\n[Baseline] Primary Model Performance:")
        print(classification_report(y_test, primary_test_pred))

        # 2. Meta-Labeling Strategy:
        # We only take trades where Primary says "Trade" (e.g. 1) AND Meta says "Correct" (1).
        # Assuming 1 is the positive class (Trade).
        # If Primary predicts 0, we don't trade anyway.
        # So effective prediction is: 1 if (Primary=1 AND Meta=1), else 0.

        # Check if Primary=1 is indeed the "Trade" signal. usually yes.
        # But wait, Primary predicts 1 (Long?) or 0 (No Trade/Short?).
        # If it's binary Price Reversal, 1 = Reversal (Buy), 0 = No Reversal (Hold/Sell).

        final_decision = (primary_test_pred == 1) & (meta_test_pred == 1)
        final_decision = final_decision.astype(int)

        print("\n[Meta-Labeling] Filtered Performance:")
        print(classification_report(y_test, final_decision))

        # Compare Precision (Confidence in taking a trade)
        prec_baseline = precision_score(
            y_test, primary_test_pred, pos_label=1, zero_division=0
        )
        prec_meta = precision_score(
            y_test, final_decision, pos_label=1, zero_division=0
        )

        print(f"Precision Improvement: {prec_baseline:.4f} -> {prec_meta:.4f}")

        return primary_final, meta_model


def main():
    # Load data
    raw_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv",
    )
    raw_data.rename(columns={VOLUME_COL: "volume", CLOSE_COL: "close"}, inplace=True)

    # Configuration
    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_splits": 3,
        "pct_embargo": 0.01,
        "use_pca": True,
        "pca_components": 0.95,
    }

    # Primary Model (XGBoost)
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
    }
    primary_model = xgb.XGBClassifier(**xgb_params)

    # Meta Model Config (AutoGluon)
    meta_config = {
        "label": "label",  # Actually we pass y directly, but adapter needs this attr
        "eval_metric": "f1",
        "presets": "medium_quality",
        "time_limit": 120,  # Higher limit for meta model as it has more features
        "path": "AutogluonModels/metalabeling",
        "verbosity": 2,
    }

    pipeline = PalazzoMetaLabelingPipeline(pipeline_config)
    pipeline.run_metalabeling_experiment(raw_data, primary_model, meta_config)


if __name__ == "__main__":
    main()
