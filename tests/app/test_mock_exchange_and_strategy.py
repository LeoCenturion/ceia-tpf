"""
Tests for MockBinanceClient (API contract) and ChronosPalazzoStrategy (end-to-end).

MockBinanceClient must produce klines with the same types as the real Binance REST API:
  [0]  open_time              int   (ms)
  [1]  open                   str
  [2]  high                   str
  [3]  low                    str
  [4]  close                  str
  [5]  volume                 str
  [6]  close_time             int   (ms)
  [7]  quote_asset_volume     str
  [8]  number_of_trades       int
[9]  taker_buy_base_vol     str
  [10] taker_buy_quote_vol    str
  [11] ignore                 str   ("0")

End-to-end strategy tests use MockBinanceClient for data; everything else is real:
ChronosPalazzoStrategy, PalazzoChronosBinaryClassificationPipeline, TimeSeriesPredictor.
The 'Naive' model is used (fit_hyperparameters) so tests run in seconds without a
Chronos model download.

Feature engineering uses SMA(20) / Aroon(14) rolling indicators — the binding cqkonstraint
for non-NaN features is SMA(20): first non-NaN feature row at index 19 (the 20th bar).
Labels drop the last bar, so 26 volume bars → 6 aligned rows → AutoGluon minimum.
"""

import inspect
import math
import os
import tempfile
import unittest

import pandas as pd

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_klines_csv(path, n_rows=10, base_ts=1577836800000):
    """Write a minimal klines CSV with constant prices (used by API contract tests)."""
    with open(path, "w") as f:
        f.write(
            "date,timestamp,open,high,low,close,volume,close_time,"
            "Volume USDT,tradeCount,taker_buy_base_asset_volume,"
            "taker_buy_quote_asset_volume,ignore\n"
        )
        for i in range(n_rows):
            ts = base_ts + i * 60_000
            ct = ts + 59_999
            f.write(
                f"2020-01-01,{ts},100.0,101.0,99.0,100.5,50.0,"
                f"{ct},5000.0,100,25.0,2500.0,0\n"
            )


def _write_ohlcv_csv(path, n_rows, volume=1.0):
    """
    Write n_rows of 1m klines with oscillating prices.

    Prices follow a sine wave + linear trend to ensure pct_change is non-zero
    and sma20_pct never equals zero (avoids division-by-zero in BB_Width_pct).
    """
    base_ts = 1_577_836_800_000
    with open(path, "w") as f:
        f.write(
            "date,timestamp,open,high,low,close,volume,close_time,"
            "Volume USDT,tradeCount,taker_buy_base_asset_volume,"
            "taker_buy_quote_asset_volume,ignore\n"
        )
        for i in range(n_rows):
            ts = base_ts + i * 60_000
            ct = ts + 59_999
            price = 100.0 + math.sin(i * 0.1) * 5.0 + i * 0.05
            high = price + 0.5
            low = price - 0.5
            f.write(
                f"2020-01-01,{ts},{price:.4f},{high:.4f},{low:.4f},"
                f"{price:.4f},{volume:.1f},{ct},{price * volume:.2f},"
                f"100,{volume / 2:.1f},{price * volume / 2:.2f},0\n"
            )


# ---------------------------------------------------------------------------
# MockBinanceClient — Binance API contract
# ---------------------------------------------------------------------------


class TestMockExchangeAPIContract(unittest.TestCase):
    """MockBinanceClient must return klines with the exact types the real API uses."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        self.tmp.close()
        _write_klines_csv(self.tmp.name, n_rows=10)

        from src.app.mock_exchange import MockBinanceClient

        self.client = MockBinanceClient(
            api_key=None, api_secret=None, mock_file=self.tmp.name
        )

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _first_kline(self, limit=1):
        return self.client.get_historical_klines("BTCUSDT", "1m", limit=limit)[0]

    # --- field count and types -----------------------------------------------

    def test_returns_12_fields_per_kline(self):
        klines = self.client.get_historical_klines("BTCUSDT", "1m", limit=5)
        for kline in klines:
            self.assertEqual(len(kline), 12)

    def test_open_time_is_int(self):
        self.assertIsInstance(self._first_kline()[0], int)

    def test_close_time_is_int(self):
        self.assertIsInstance(self._first_kline()[6], int)

    def test_number_of_trades_is_int(self):
        self.assertIsInstance(self._first_kline()[8], int)

    def test_ohlcv_and_quote_fields_are_strings(self):
        """Fields 1-5 and 7, 9, 10, 11 must be strings per Binance REST API spec."""
        kline = self._first_kline()
        for idx in (1, 2, 3, 4, 5, 7, 9, 10, 11):
            self.assertIsInstance(
                kline[idx],
                str,
                f"Field [{idx}] should be str, got {type(kline[idx]).__name__}",
            )

    def test_ignore_field_is_string_zero(self):
        self.assertEqual(self._first_kline()[11], "0")

    # --- pagination / exhaustion ---------------------------------------------

    def test_limit_respected(self):
        klines = self.client.get_historical_klines("BTCUSDT", "1m", limit=3)
        self.assertEqual(len(klines), 3)

    def test_returns_empty_list_when_data_exhausted(self):
        # The mock advances current_index by 1 per call (simulates time ticking
        # forward one candle at a time), so we need data_length calls to exhaust it.
        for _ in range(self.client.data_length):
            self.client.get_historical_klines("BTCUSDT", "1m", limit=1)
        result = self.client.get_historical_klines("BTCUSDT", "1m", limit=1)
        self.assertEqual(result, [])

    # --- timestamp normalisation ---------------------------------------------

    def test_microsecond_timestamps_normalised_to_milliseconds(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp.close()
        with open(tmp.name, "w") as f:
            f.write(
                "date,timestamp,open,high,low,close,volume,close_time,"
                "Volume USDT,tradeCount,taker_buy_base_asset_volume,"
                "taker_buy_quote_asset_volume,ignore\n"
            )
            ms_ts = 1_577_836_800_000
            f.write(
                f"2020-01-01,{ms_ts * 1000},100.0,101.0,99.0,100.5,50.0,"
                f"{(ms_ts + 59_999) * 1000},5000.0,100,25.0,2500.0,0\n"
            )

        from src.app.mock_exchange import MockBinanceClient

        client = MockBinanceClient(api_key=None, api_secret=None, mock_file=tmp.name)
        kline = client.get_historical_klines("BTCUSDT", "1m", limit=1)[0]
        os.unlink(tmp.name)

        # Must be in ms range (year ~2020), not µs range (year ~57000)
        self.assertEqual(kline[0], ms_ts)
        self.assertLess(kline[0], 2_000_000_000_000)  # < year 2033 in ms

    # --- interface parity with BinanceClient ---------------------------------

    def test_signature_matches_real_client(self):
        """MockBinanceClient.get_historical_klines must accept the same parameters."""
        from src.app.exchange import BinanceClient
        from src.app.mock_exchange import MockBinanceClient

        mock_params = list(
            inspect.signature(MockBinanceClient.get_historical_klines).parameters.keys()
        )[1:]  # drop 'self'
        real_params = list(
            inspect.signature(BinanceClient.get_historical_klines).parameters.keys()
        )[1:]

        self.assertEqual(mock_params, real_params)


# ---------------------------------------------------------------------------
# ChronosPalazzoStrategy — end-to-end with MockBinanceClient
# ---------------------------------------------------------------------------


class TestChronosPalazzoStrategyEndToEnd(unittest.TestCase):
    """
    End-to-end tests for ChronosPalazzoStrategy.

    MockBinanceClient replaces the real Binance exchange; everything else is real:
      - ChronosPalazzoStrategy (volume bar accumulation, signal generation)
      - PalazzoChronosBinaryClassificationPipeline (feature engineering + labeling)
      - TimeSeriesPredictor with Naive model (fast, no Chronos download needed)

    Volume bars: threshold=2, volume=1.0/row → 1 bar per 2 rows.

    Feature engineering uses SMA(20) rolling indicators. The binding constraint for
    non-NaN features is SMA(20): bar index 19 is the first non-NaN feature row.
    Labels drop the last bar → 26 bars yields 6 aligned rows (AutoGluon minimum).
    """

    _CONFIG = {
        "volume_threshold": 2,
        "min_bars_to_fit": 3,
        "refit_every_n_bars": 15,
        "prediction_length": 1,
        "fit_hyperparameters": {"Naive": {}},
    }

    @staticmethod
    def _klines_to_df(klines):
        """Convert raw klines list to a timestamped DataFrame, mirroring bot.py."""
        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        for col in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ):
            df[col] = pd.to_numeric(df[col])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.set_index("timestamp")

    def _make_client(self, path):
        from src.app.mock_exchange import MockBinanceClient

        return MockBinanceClient(api_key=None, api_secret=None, mock_file=path)

    def _make_strategy(self, extra_config=None):
        from src.app.chronos_strategy import ChronosPalazzoStrategy

        return ChronosPalazzoStrategy({**self._CONFIG, **(extra_config or {})})

    # -------------------------------------------------------------------------
    # Case 1 — not enough data to form a volume bar
    # -------------------------------------------------------------------------

    def test_case1_no_volume_bar_formed(self):
        """
        Cumulative volume never reaches threshold → no bar emitted, signal is None,
        volume_bars is empty.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        tmp.close()
        try:
            _write_ohlcv_csv(tmp.name, n_rows=5, volume=1.0)
            client = self._make_client(tmp.name)
            strategy = self._make_strategy({"volume_threshold": 100})

            klines = client.get_historical_klines("BTCUSDT", "1m", limit=5)
            signal = strategy.get_signal(self._klines_to_df(klines))

            self.assertEqual(signal, "HOLD")
            self.assertEqual(len(strategy.volume_bars), 0)
        finally:
            os.unlink(tmp.name)

    # -------------------------------------------------------------------------
    # Case 2 — volume bars are formed
    # -------------------------------------------------------------------------

    def test_case2_volume_bars_are_formed(self):
        """
        Cumulative volume crosses threshold → volume bars are created with the
        expected OHLCV columns. Signal may be None (not enough bars yet to train),
        but bars must exist.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        tmp.close()
        try:
            _write_ohlcv_csv(tmp.name, n_rows=50, volume=1.0)  # → 25 bars
            client = self._make_client(tmp.name)
            strategy = self._make_strategy()

            klines = client.get_historical_klines("BTCUSDT", "1m", limit=50)
            strategy.get_signal(self._klines_to_df(klines))

            self.assertGreater(len(strategy.volume_bars), 0)
            bar = strategy.volume_bars.iloc[0]
            for col in (
                "open_price",
                "close_price",
                "High",
                "Low",
                "total_volume",
                "bar_return",
            ):
                self.assertIn(col, bar.index, f"Volume bar missing column '{col}'")
        finally:
            os.unlink(tmp.name)

    # -------------------------------------------------------------------------
    # Case 3 — not enough bars to train the model
    # -------------------------------------------------------------------------

    def test_case3_not_enough_bars_to_train(self):
        """
        Volume bars are formed but SMA(20) / Bollinger Bands (20) warm-up means
        all feature rows are NaN until bar 19. With only 10 bars the pipeline
        returns 0 non-NaN feature rows → min_bars_to_fit check fails → signal is
        None, model_is_fit stays False.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        tmp.close()
        try:
            _write_ohlcv_csv(tmp.name, n_rows=20, volume=1.0)  # → 10 bars
            client = self._make_client(tmp.name)
            strategy = self._make_strategy()

            klines = client.get_historical_klines("BTCUSDT", "1m", limit=20)
            signal = strategy.get_signal(self._klines_to_df(klines))

            self.assertGreater(
                len(strategy.volume_bars),
                0,
                "Bars should have been formed even if model cannot train",
            )
            self.assertEqual(signal, "HOLD")
            self.assertFalse(strategy.model_is_fit)
        finally:
            os.unlink(tmp.name)

    # -------------------------------------------------------------------------
    # Case 4 — model trains once and predicts
    # -------------------------------------------------------------------------

    def test_case4_model_trains_once_and_predicts(self):
        """
        200 volume bars → pipeline produces ~175 aligned feature rows, well above
        AutoGluon's 6-observation minimum → model fits, signal is 'BUY' or 'SELL'.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        tmp.close()
        try:
            _write_ohlcv_csv(tmp.name, n_rows=400, volume=1.0)  # → 200 bars
            client = self._make_client(tmp.name)
            strategy = self._make_strategy()

            klines = client.get_historical_klines("BTCUSDT", "1m", limit=400)
            signal = strategy.get_signal(self._klines_to_df(klines))

            self.assertTrue(strategy.model_is_fit)
            self.assertEqual(strategy.bars_since_refit, 0)
            self.assertIn(signal, ("BUY", "SELL"))
        finally:
            os.unlink(tmp.name)

    # -------------------------------------------------------------------------
    # Case 5 — model trains twice and predicts
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Case 6 — prediction_length=2 does not raise known_covariates error
    # -------------------------------------------------------------------------

    def test_case6_prediction_length_2_produces_signal(self):
        """
        Regression test: prediction_length=2 requires known_covariates to cover
        2 future timesteps after the end of train_data. Without the fix, AutoGluon
        raises: 'known_covariates should include the item_id and timestamp values
        covering the forecast horizon'.

        The production default is prediction_length=2; the other tests all use 1,
        which is why this bug was never caught.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        tmp.close()
        try:
            _write_ohlcv_csv(tmp.name, n_rows=400, volume=1.0)  # → 200 volume bars
            client = self._make_client(tmp.name)
            strategy = self._make_strategy({"prediction_length": 2})

            klines = client.get_historical_klines("BTCUSDT", "1m", limit=400)
            signal = strategy.get_signal(self._klines_to_df(klines))

            self.assertTrue(
                strategy.model_is_fit, "Model should have fit with 200 bars"
            )
            self.assertIn(
                signal, ("BUY", "SELL"), f"Expected BUY or SELL, got: {signal!r}"
            )
        finally:
            os.unlink(tmp.name)

    def test_case5_model_trains_twice_and_predicts(self):
        """
        Two calls to get_signal with non-overlapping new data trigger two separate
        model fits. The second fit is the refit path (bars_since_refit ≥
        refit_every_n_bars). After the second fit bars_since_refit resets to 0
        and the signal is still valid.

        MockBinanceClient sliding-window mechanics:
          call 1 (limit=300): rows 0–299  → 150 bars → initial train
          call 2 (limit=350): rows 1–350  → new rows 300–350 = 51 rows = 25 new bars
                                            25 ≥ refit_every_n_bars(15) → refit
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        tmp.close()
        try:
            _write_ohlcv_csv(tmp.name, n_rows=600, volume=1.0)
            client = self._make_client(tmp.name)
            strategy = self._make_strategy()

            # First call: rows 0–299 → 150 bars → initial train
            klines1 = client.get_historical_klines("BTCUSDT", "1m", limit=300)
            signal1 = strategy.get_signal(self._klines_to_df(klines1))
            self.assertTrue(strategy.model_is_fit)
            self.assertEqual(strategy.bars_since_refit, 0)
            self.assertIn(signal1, ("BUY", "SELL"))
            bars_after_first = len(strategy.volume_bars)

            # Second call: current_index=1, limit=350 → rows 1–350
            # strategy skips rows ≤ last_ts (row 299); new rows 300–350 → 25 bars
            klines2 = client.get_historical_klines("BTCUSDT", "1m", limit=350)
            signal2 = strategy.get_signal(self._klines_to_df(klines2))

            self.assertGreater(
                len(strategy.volume_bars),
                bars_after_first,
                "New bars must have been formed in the second call",
            )
            self.assertEqual(
                strategy.bars_since_refit,
                0,
                "bars_since_refit must reset to 0 after refit",
            )
            self.assertIn(signal2, ("BUY", "SELL"))
        finally:
            os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
