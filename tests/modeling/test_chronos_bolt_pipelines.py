"""
E2E wiring tests for Chronos-Bolt pipeline variants.

PalazzoChronosBoltBinaryClassificationPipeline  — TimeSeriesPredictor + Bolt model.
ChronosBoltFeaturePipeline                       — Bolt embeddings as extra features
                                                   fed into PalazzoXGBoostPipeline.
Heavy ML components are mocked so the suite runs without GPU or downloads.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_volume_bars(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.005, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "open_price": open_,
            "High": high,
            "Low": low,
            "close_price": close,
            "total_volume": rng.uniform(50000, 200000, n),
            "intra_bar_std": rng.uniform(0.0001, 0.005, n),
            "bar_return": np.diff(close, prepend=close[0]) / close,
        },
        index=idx,
    )


_EMBED_DIM = 8


def _make_mock_bolt_pipeline(embed_dim: int = _EMBED_DIM) -> MagicMock:
    mock_bolt = MagicMock()

    def fake_embed(ts_tensor):
        emb = MagicMock()
        (
            emb.mean.return_value
                .squeeze.return_value
                .float.return_value
                .cpu.return_value
                .numpy.return_value
        ) = np.ones(embed_dim, dtype=np.float32)
        return emb, (MagicMock(), MagicMock())

    mock_bolt.embed.side_effect = fake_embed
    return mock_bolt


# ---------------------------------------------------------------------------
# PalazzoChronosBoltBinaryClassificationPipeline
# ---------------------------------------------------------------------------

class TestChronosBoltPalazzoPipeline(unittest.TestCase):

    def setUp(self):
        from src.modeling.chronos_bolt_pipeline_palazzo import (
            PalazzoChronosBoltBinaryClassificationPipeline,
        )
        self.config = {
            "volume_threshold": 50000,
            "chronos_model": "autogluon/chronos-bolt-small",
            "prediction_length": 2,
        }
        self.pipeline = PalazzoChronosBoltBinaryClassificationPipeline(self.config)
        self.bars = _make_volume_bars(80)

    def test_default_model_is_chronos_bolt(self):
        from src.modeling.chronos_bolt_pipeline_palazzo import (
            PalazzoChronosBoltBinaryClassificationPipeline,
        )
        p = PalazzoChronosBoltBinaryClassificationPipeline({})
        self.assertIn("chronos-bolt", p.config["chronos_model"])

    def test_step_2_returns_dataframe_without_nan(self):
        features = self.pipeline.step_2_feature_engineering(self.bars.copy())
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
        self.assertFalse(features.isnull().any().any())

    def test_step_3_returns_binary_labels(self):
        y, _, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        self.assertIsInstance(y, pd.Series)
        self.assertTrue(set(y.unique()).issubset({1.0, -1.0}))

    def test_step_3_label_length_is_bars_minus_one(self):
        y, _, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        self.assertEqual(len(y), len(self.bars) - 1)

    @patch("src.modeling.chronos_bolt_pipeline_palazzo.TimeSeriesPredictor")
    def test_fit_predictor_builds_timeseries_dataframe(self, MockPredictor):
        features = self.pipeline.step_2_feature_engineering(self.bars.copy())
        y, _, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        common = features.index.intersection(y.index)
        features, y = features.loc[common], y.loc[common]

        with patch("os.path.exists", return_value=False):
            self.pipeline.fit_predictor(features, y)

        from autogluon.timeseries import TimeSeriesDataFrame
        ts_arg = MockPredictor.return_value.fit.call_args[0][0]
        self.assertIsInstance(ts_arg, TimeSeriesDataFrame)

    @patch("src.modeling.chronos_bolt_pipeline_palazzo.TimeSeriesPredictor")
    def test_fit_predictor_passes_bolt_model_path(self, MockPredictor):
        features = self.pipeline.step_2_feature_engineering(self.bars.copy())
        y, _, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        common = features.index.intersection(y.index)
        features, y = features.loc[common], y.loc[common]

        with patch("os.path.exists", return_value=False):
            self.pipeline.fit_predictor(features, y)

        hparams = MockPredictor.return_value.fit.call_args[1]["hyperparameters"]
        self.assertIn("chronos-bolt", hparams["Chronos"]["model_path"])

    @patch("src.modeling.chronos_bolt_pipeline_palazzo.TimeSeriesPredictor")
    def test_fit_predictor_known_covariates_excludes_reserved(self, MockPredictor):
        features = self.pipeline.step_2_feature_engineering(self.bars.copy())
        y, _, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        common = features.index.intersection(y.index)
        features, y = features.loc[common], y.loc[common]

        with patch("os.path.exists", return_value=False):
            _, known_cov_names = self.pipeline.fit_predictor(features, y)

        for reserved in ("target", "item_id", "timestamp"):
            self.assertNotIn(reserved, known_cov_names)
        self.assertGreater(len(known_cov_names), 0)

    @patch("src.modeling.chronos_bolt_pipeline_palazzo.TimeSeriesPredictor")
    def test_predict_next_positive_mean_returns_1(self, MockPredictor):
        features = self.pipeline.step_2_feature_engineering(self.bars.copy())
        y, _, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        common = features.index.intersection(y.index)
        features, y = features.loc[common], y.loc[common]

        MockPredictor.return_value.predict.return_value.loc.__getitem__ = MagicMock(
            return_value=pd.DataFrame({"mean": [2.0]})
        )
        with patch("os.path.exists", return_value=False):
            predictor, known_cov_names = self.pipeline.fit_predictor(features, y)

        self.assertEqual(
            self.pipeline.predict_next(predictor, features, y, known_cov_names), 1
        )

    @patch("src.modeling.chronos_bolt_pipeline_palazzo.TimeSeriesPredictor")
    def test_predict_next_negative_mean_returns_0(self, MockPredictor):
        features = self.pipeline.step_2_feature_engineering(self.bars.copy())
        y, _, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        common = features.index.intersection(y.index)
        features, y = features.loc[common], y.loc[common]

        MockPredictor.return_value.predict.return_value.loc.__getitem__ = MagicMock(
            return_value=pd.DataFrame({"mean": [-1.0]})
        )
        with patch("os.path.exists", return_value=False):
            predictor, known_cov_names = self.pipeline.fit_predictor(features, y)

        self.assertEqual(
            self.pipeline.predict_next(predictor, features, y, known_cov_names), 0
        )


# ---------------------------------------------------------------------------
# ChronosBoltFeaturePipeline  (Bolt embeddings → XGBoost)
# ---------------------------------------------------------------------------

class TestChronosBoltFeaturePipeline(unittest.TestCase):

    def setUp(self):
        from src.modeling.chronos_bolt_feature_pipeline import ChronosBoltFeaturePipeline
        self.config = {
            "volume_threshold": 50000,
            "tau": 0.7,
            "chronos_model_name": "autogluon/chronos-bolt-tiny",
            "chronos_window_size": 16,
            "chronos_stride": 1,
        }
        self.pipeline = ChronosBoltFeaturePipeline(self.config)
        self.pipeline.bolt_pipeline = _make_mock_bolt_pipeline()
        self.bars = _make_volume_bars(60)

    # --- feature engineering ---

    def test_step_2_includes_bolt_embed_columns(self):
        features = self.pipeline.step_2_feature_engineering(self.bars.copy())
        bolt_cols = [c for c in features.columns if c.startswith("bolt_embed_")]
        self.assertGreater(len(bolt_cols), 0)

    def test_step_2_includes_tabular_columns(self):
        features = self.pipeline.step_2_feature_engineering(self.bars.copy())
        tabular_cols = [c for c in features.columns if not c.startswith("bolt_embed_")]
        self.assertGreater(len(tabular_cols), 0)

    def test_step_2_no_nan(self):
        features = self.pipeline.step_2_feature_engineering(self.bars.copy())
        self.assertFalse(features.isnull().any().any())

    def test_step_2_output_index_subset_of_bars(self):
        features = self.pipeline.step_2_feature_engineering(self.bars.copy())
        self.assertTrue(features.index.isin(self.bars.index).all())

    def test_step_2_embed_called_once_per_window(self):
        mock_bolt = _make_mock_bolt_pipeline()
        self.pipeline.bolt_pipeline = mock_bolt
        window_size = self.config["chronos_window_size"]

        # expected window count is against the tabular-feature-aligned bars
        from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
        tabular = PalazzoXGBoostPipeline(self.config).step_2_feature_engineering(
            self.bars.copy()
        )
        expected = max(0, len(tabular) - window_size + 1)

        self.pipeline.step_2_feature_engineering(self.bars.copy())
        self.assertEqual(mock_bolt.embed.call_count, expected)

    def test_step_2_lazy_loads_bolt_pipeline(self):
        self.pipeline.bolt_pipeline = None

        with patch(
            "src.modeling.chronos_bolt_feature_pipeline.ChronosBoltPipeline",
            create=True,
        ) as MockBolt:
            MockBolt.from_pretrained.return_value = _make_mock_bolt_pipeline()
            self.pipeline.step_2_feature_engineering(self.bars.copy())

        MockBolt.from_pretrained.assert_called_once()
        self.assertIn("chronos-bolt", MockBolt.from_pretrained.call_args[0][0])

    def test_step_2_uses_configured_model_name(self):
        self.pipeline.bolt_pipeline = None

        with patch(
            "src.modeling.chronos_bolt_feature_pipeline.ChronosBoltPipeline",
            create=True,
        ) as MockBolt:
            MockBolt.from_pretrained.return_value = _make_mock_bolt_pipeline()
            self.pipeline.step_2_feature_engineering(self.bars.copy())

        self.assertEqual(
            MockBolt.from_pretrained.call_args[0][0], "autogluon/chronos-bolt-tiny"
        )

    def test_step_2_reuses_bolt_pipeline_across_calls(self):
        mock_bolt = _make_mock_bolt_pipeline()
        self.pipeline.bolt_pipeline = mock_bolt

        self.pipeline.step_2_feature_engineering(self.bars.copy())
        self.pipeline.step_2_feature_engineering(self.bars.copy())

        self.assertIs(self.pipeline.bolt_pipeline, mock_bolt)

    # --- labeling (from PalazzoXGBoostPipeline) ---

    def test_step_3_returns_series(self):
        y, weights, t1 = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        self.assertIsInstance(y, pd.Series)
        self.assertIsInstance(weights, pd.Series)

    def test_step_3_labels_are_binary(self):
        y, _, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        self.assertTrue(set(y.unique()).issubset({0, 1}))

    # --- no TimeSeriesPredictor dependency ---

    def test_pipeline_has_no_fit_predictor(self):
        self.assertFalse(hasattr(self.pipeline, "fit_predictor"))

    def test_pipeline_has_no_predict_next(self):
        self.assertFalse(hasattr(self.pipeline, "predict_next"))


if __name__ == "__main__":
    unittest.main()
