"""
E2E wiring tests for ChronosBoltFeaturePipeline.

Adds Chronos-Bolt patch embeddings to the PalazzoXGBoostPipeline tabular
feature set; downstream model is XGBoost (same as the parent pipeline).
Heavy ML components are mocked so the suite runs without GPU or downloads.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


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


class TestChronosBoltFeaturePipeline(unittest.TestCase):

    def setUp(self):
        from src.modeling.transformers.chronos_bolt_feature_pipeline import ChronosBoltFeaturePipeline
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

        from src.modeling.machine_learning.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
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

    # --- labeling from PalazzoXGBoostPipeline ---

    def test_step_3_returns_series_with_sample_weights(self):
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
