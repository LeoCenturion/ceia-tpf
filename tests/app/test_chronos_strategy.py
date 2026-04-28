import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.app.chronos_strategy import ChronosPalazzoStrategy


class TestChronosPalazzoStrategy(unittest.TestCase):
    def setUp(self):
        self.config = {"volume_threshold": 50000, "min_bars_to_fit": 30}

    def _make_strategy(self):
        with patch(
            "src.app.chronos_strategy.PalazzoChronosBinaryClassificationPipeline"
        ):
            return ChronosPalazzoStrategy(self.config)

    def _setup_pipeline_mocks(self, strategy, predicted_class=1):
        n = 35  # > min_bars_to_fit=30
        idx = range(n)
        features = pd.DataFrame({"f": [1.0] * n}, index=idx)
        y = pd.Series([1] * n, index=idx)
        mock_predictor = MagicMock()

        strategy.pipeline.step_2_feature_engineering = MagicMock(return_value=features)
        strategy.pipeline.step_3_labeling_and_weighting = MagicMock(return_value=(y,))
        strategy.pipeline.fit_predictor = MagicMock(
            return_value=(mock_predictor, ["f"])
        )
        strategy.pipeline.predict_next = MagicMock(return_value=predicted_class)

    def _make_sample_data(self):
        return pd.DataFrame(
            {
                "close": [100],
                "open": [90],
                "high": [110],
                "low": [80],
                "volume": [1000],
            },
            index=pd.to_datetime(["2023-01-01"]),
        )

    def test_initialization_creates_pipeline(self):
        with patch(
            "src.app.chronos_strategy.PalazzoChronosBinaryClassificationPipeline"
        ) as mock_cls:
            strategy = ChronosPalazzoStrategy(self.config)
            mock_cls.assert_called_once_with(self.config)
            self.assertIs(strategy.pipeline, mock_cls.return_value)

    def test_get_signal_buy(self):
        strategy = self._make_strategy()
        self._setup_pipeline_mocks(strategy, predicted_class=1)

        with patch.object(strategy, "_process_new_1m_bars", return_value=True):
            signal = strategy.get_signal(self._make_sample_data())

        self.assertEqual(signal, "BUY")

    def test_get_signal_sell(self):
        strategy = self._make_strategy()
        self._setup_pipeline_mocks(strategy, predicted_class=0)

        with patch.object(strategy, "_process_new_1m_bars", return_value=True):
            signal = strategy.get_signal(self._make_sample_data())

        self.assertEqual(signal, "SELL")

    def test_get_signal_no_new_bars_returns_hold(self):
        strategy = self._make_strategy()

        with patch.object(strategy, "_process_new_1m_bars", return_value=False):
            signal = strategy.get_signal(self._make_sample_data())

        self.assertEqual(signal, "HOLD")

    def test_get_signal_prediction_error_returns_hold(self):
        strategy = self._make_strategy()
        self._setup_pipeline_mocks(strategy)
        strategy.pipeline.step_2_feature_engineering.side_effect = Exception(
            "Pipeline failed"
        )

        with (
            patch.object(strategy, "_process_new_1m_bars", return_value=True),
            patch("logging.error") as mock_log_error,
        ):
            signal = strategy.get_signal(self._make_sample_data())

        self.assertEqual(signal, "HOLD")
        mock_log_error.assert_called_once()


class TestWarmup(unittest.TestCase):
    """
    volume_threshold=100, each 1m bar has volume=30.
    Accumulation: 30 → 60 → 90 → 120 (≥100) → volume bar created, reset.
    So every 4 bars produces one complete volume bar.
    10 bars → 2 complete bars + 2 bars left as partial state (cumulative=60).
    """

    _THRESHOLD = 100
    _VOL_PER_BAR = 30  # 4 bars needed per volume bar

    def setUp(self):
        self.config = {
            "volume_threshold": self._THRESHOLD,
            "min_bars_to_fit": 30,
        }

    def _make_strategy(self):
        with patch(
            "src.app.chronos_strategy.PalazzoChronosBinaryClassificationPipeline"
        ):
            return ChronosPalazzoStrategy(self.config)

    def _write_csv(self, timestamps) -> str:
        df = pd.DataFrame(
            {
                "date": timestamps,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": float(self._VOL_PER_BAR),
            }
        )
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.to_csv(f.name, index=False)
        f.close()
        return f.name

    def _live_bars(self, timestamps) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.0,
                "volume": float(self._VOL_PER_BAR),
            },
            index=pd.to_datetime(timestamps),
        )

    def tearDown(self):
        # clean up any temp files created during the test
        pass

    # ------------------------------------------------------------------
    # No-gap: live data starts immediately after the CSV ends
    # ------------------------------------------------------------------

    def test_no_gap_volume_bars_built_from_csv(self):
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1min")
        path = self._write_csv(timestamps)
        try:
            strategy = self._make_strategy()
            strategy._warmup_from_csv(path, lookback=None)
            self.assertEqual(len(strategy.volume_bars), 2)
        finally:
            os.unlink(path)

    def test_no_gap_partial_accumulator_reset_after_warmup(self):
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1min")
        path = self._write_csv(timestamps)
        try:
            strategy = self._make_strategy()
            strategy._warmup_from_csv(path, lookback=None)
            self.assertEqual(strategy.current_bar_data, [])
            self.assertEqual(strategy.cumulative_volume, 0.0)
        finally:
            os.unlink(path)

    def test_no_gap_last_processed_timestamp_set_to_csv_end(self):
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1min")
        path = self._write_csv(timestamps)
        try:
            strategy = self._make_strategy()
            strategy._warmup_from_csv(path, lookback=None)
            self.assertEqual(strategy.last_processed_timestamp, timestamps[-1])
        finally:
            os.unlink(path)

    def test_no_gap_csv_bars_not_reprocessed_on_live_tick(self):
        """Live data that overlaps with the CSV must not produce extra volume bars."""
        csv_timestamps = pd.date_range("2024-01-01", periods=10, freq="1min")
        path = self._write_csv(csv_timestamps)
        try:
            strategy = self._make_strategy()
            strategy._warmup_from_csv(path, lookback=None)
            bars_after_warmup = len(strategy.volume_bars)

            # live tick delivers the last 4 CSV bars again + 4 new bars
            overlap = csv_timestamps[-4:]
            new = pd.date_range(
                csv_timestamps[-1] + pd.Timedelta("1min"), periods=4, freq="1min"
            )
            live = self._live_bars(overlap.tolist() + new.tolist())
            strategy._process_new_1m_bars(live)

            # only the 4 new bars should be processed → 1 new volume bar
            self.assertEqual(len(strategy.volume_bars), bars_after_warmup + 1)
        finally:
            os.unlink(path)

    # ------------------------------------------------------------------
    # Gap: live data starts well after the CSV ends
    # ------------------------------------------------------------------

    def test_gap_live_bars_processed_after_gap(self):
        """Bars arriving after a gap must not be filtered out."""
        csv_timestamps = pd.date_range("2024-01-01", periods=10, freq="1min")
        path = self._write_csv(csv_timestamps)
        try:
            strategy = self._make_strategy()
            strategy._warmup_from_csv(path, lookback=None)
            bars_after_warmup = len(strategy.volume_bars)

            # live data starts 1 day after the CSV ends
            gap_start = csv_timestamps[-1] + pd.Timedelta("1d")
            live_timestamps = pd.date_range(gap_start, periods=4, freq="1min")
            strategy._process_new_1m_bars(self._live_bars(live_timestamps))

            self.assertEqual(len(strategy.volume_bars), bars_after_warmup + 1)
        finally:
            os.unlink(path)

    def test_gap_first_live_volume_bar_uses_live_open_price(self):
        """After a gap the open of the first new volume bar must come from live data, not stale CSV data."""
        csv_timestamps = pd.date_range("2024-01-01", periods=10, freq="1min")
        path = self._write_csv(csv_timestamps)
        try:
            strategy = self._make_strategy()
            strategy._warmup_from_csv(path, lookback=None)

            gap_start = csv_timestamps[-1] + pd.Timedelta("1d")
            live_timestamps = pd.date_range(gap_start, periods=4, freq="1min")
            strategy._process_new_1m_bars(self._live_bars(live_timestamps))

            first_live_bar = strategy.volume_bars.iloc[-1]
            # live bars have open=200; CSV bars have open=100
            self.assertEqual(first_live_bar["open_price"], 200.0)
        finally:
            os.unlink(path)

    def test_gap_partial_state_not_carried_across_gap(self):
        """The partial accumulator must be empty before any live bar is processed."""
        csv_timestamps = pd.date_range("2024-01-01", periods=10, freq="1min")
        path = self._write_csv(csv_timestamps)
        try:
            strategy = self._make_strategy()
            strategy._warmup_from_csv(path, lookback=None)

            # Verify the accumulator is clean before live data arrives
            self.assertEqual(strategy.current_bar_data, [])
            self.assertEqual(strategy.cumulative_volume, 0.0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
