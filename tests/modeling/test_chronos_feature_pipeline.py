"""
End-to-end tests for ChronosFeaturePipeline.

No mocking — exercises the full path through actual Chronos T5 model loading,
tokenisation, encoder embedding extraction, and concatenation with tabular
features.  Uses synthetic volume bars and a tiny window size to minimise
compute time.

Requires amazon/chronos-t5-tiny to be present in the HuggingFace cache
(or network access on first run).
"""

import unittest

import numpy as np
import pandas as pd

from src.modeling.chronos_feature_pipeline import ChronosFeaturePipeline


def _make_volume_bars(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
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


class TestChronosFeaturePipelineE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = {
            "volume_threshold": 50000,
            "tau": 0.7,
            "n_splits": 2,
            "pct_embargo": 0.01,
            "use_pca": False,
            "chronos_model_name": "amazon/chronos-t5-tiny",
            "chronos_window_size": 8,
            "chronos_stride": 1,
        }
        cls.pipeline = ChronosFeaturePipeline(cls.config)
        cls.bars = _make_volume_bars(80)
        # Pre-warm: loads + caches the model for the rest of the suite.
        cls.features = cls.pipeline.step_2_feature_engineering(cls.bars.copy())

    # ------------------------------------------------------------------ #
    # step_2 output structure                                              #
    # ------------------------------------------------------------------ #

    def test_step_2_returns_dataframe(self):
        self.assertIsInstance(self.features, pd.DataFrame)

    def test_step_2_non_empty(self):
        self.assertGreater(len(self.features), 0)

    def test_step_2_has_chronos_embed_columns(self):
        embed_cols = [c for c in self.features.columns if c.startswith("chronos_embed_")]
        self.assertGreater(len(embed_cols), 0)

    def test_step_2_has_tabular_feature_columns(self):
        tabular_cols = [c for c in self.features.columns if c.startswith("feature_")]
        self.assertGreater(len(tabular_cols), 0)

    def test_step_2_no_nan(self):
        self.assertFalse(self.features.isnull().any().any())

    def test_step_2_no_inf(self):
        numeric = self.features.select_dtypes(include="number")
        self.assertFalse(np.isinf(numeric.values).any())

    def test_step_2_output_index_is_subset_of_bars_index(self):
        self.assertTrue(self.features.index.isin(self.bars.index).all())

    def test_step_2_row_count_bounded_by_window_constraint(self):
        # Rows ≤ len(bars) - window_size + 1 after all NaN-drops.
        max_possible = len(self.bars) - self.config["chronos_window_size"] + 1
        self.assertLessEqual(len(self.features), max_possible)

    # ------------------------------------------------------------------ #
    # Chronos embedding quality                                            #
    # ------------------------------------------------------------------ #

    def test_embed_columns_have_consistent_dimension(self):
        embed_cols = [c for c in self.features.columns if c.startswith("chronos_embed_")]
        # All embed columns must be indexed from 0 to d_model-1 with no gaps.
        indices = sorted(int(c.split("_")[-1]) for c in embed_cols)
        self.assertEqual(indices, list(range(len(embed_cols))))

    def test_embed_values_are_finite_floats(self):
        embed_cols = [c for c in self.features.columns if c.startswith("chronos_embed_")]
        arr = self.features[embed_cols].values
        self.assertTrue(np.isfinite(arr).all())

    def test_embed_values_vary_across_rows(self):
        # Embeddings should not be identical for every window (model is doing work).
        embed_cols = [c for c in self.features.columns if c.startswith("chronos_embed_")]
        arr = self.features[embed_cols].values
        row_stds = arr.std(axis=0)
        self.assertTrue((row_stds > 0).any())

    # ------------------------------------------------------------------ #
    # Model caching                                                        #
    # ------------------------------------------------------------------ #

    def test_chronos_model_loaded_after_step_2(self):
        self.assertIsNotNone(self.pipeline.chronos_model)

    def test_chronos_tokenizer_loaded_after_step_2(self):
        self.assertIsNotNone(self.pipeline.chronos_tokenizer)

    def test_second_call_reuses_same_model_object(self):
        model_before = self.pipeline.chronos_model
        self.pipeline.step_2_feature_engineering(self.bars.copy())
        self.assertIs(self.pipeline.chronos_model, model_before)

    # ------------------------------------------------------------------ #
    # step_3 (inherited from PalazzoXGBoostPipeline)                      #
    # ------------------------------------------------------------------ #

    def test_step_3_returns_series_labels_and_weights(self):
        y, weights, t1 = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        self.assertIsInstance(y, pd.Series)
        self.assertIsInstance(weights, pd.Series)
        self.assertIsInstance(t1, pd.Series)

    def test_step_3_labels_are_binary(self):
        y, _, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        self.assertTrue(set(y.unique()).issubset({0, 1}))

    def test_step_3_sample_weights_are_positive(self):
        _, weights, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        self.assertTrue((weights > 0).all())

    # ------------------------------------------------------------------ #
    # Features + labels can be aligned for downstream training            #
    # ------------------------------------------------------------------ #

    def test_features_and_labels_share_common_index(self):
        y, _, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        common = self.features.index.intersection(y.index)
        self.assertGreater(len(common), 0)

    def test_aligned_features_have_no_nan(self):
        y, _, _ = self.pipeline.step_3_labeling_and_weighting(self.bars.copy())
        common = self.features.index.intersection(y.index)
        self.assertFalse(self.features.loc[common].isnull().any().any())


if __name__ == "__main__":
    unittest.main()
