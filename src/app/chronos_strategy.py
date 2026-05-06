import logging
from typing import Literal, Any, Optional

import numpy as np
import pandas as pd

from src.modeling.transformers.chronos_pipeline_palazzo import (
    PalazzoChronosBinaryClassificationPipeline,
)
from src.app.strategy import Strategy


class ChronosPalazzoStrategy(Strategy):
    def __init__(self, config: dict[str, Any]):
        self.pipeline = PalazzoChronosBinaryClassificationPipeline(config)

        # Volume bar settings
        self.volume_threshold: float = config.get("volume_threshold", 50000)
        self.current_bar_data: list[pd.Series] = []
        self.cumulative_volume: float = 0.0
        self.volume_bars: pd.DataFrame = pd.DataFrame()
        self.last_processed_timestamp: Optional[pd.Timestamp] = None

        # Model refitting settings
        self.refit_every_n_bars: int = config.get("refit_every_n_bars", 10)
        self.bars_since_refit: int = 0
        self.model_is_fit: bool = False
        self.min_bars_to_fit: int = config.get("min_bars_to_fit", 30)

        # Predictor state (populated by pipeline.fit_predictor)
        self.predictor: Any = None
        self.known_covariates_names: Optional[list[str]] = None

        warmup_csv: Optional[str] = config.get("warmup_csv")
        if warmup_csv:
            warmup_lookback: Optional[int] = config.get("warmup_lookback")
            self._warmup_from_csv(warmup_csv, warmup_lookback)

    def _warmup_from_csv(self, path: str, lookback: Optional[int]) -> None:
        logging.info(f"Warming up from historical data: {path}")
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df.index = pd.to_datetime(df.index)
        df = df[["open", "high", "low", "close", "volume"]]
        if lookback is not None:
            df = df.iloc[-lookback:]
        self._process_new_1m_bars(df)
        discarded = len(self.current_bar_data)
        self.current_bar_data = []
        self.cumulative_volume = 0.0
        logging.info(
            f"Warmup complete: {len(self.volume_bars)} volume bars built "
            f"from {len(df)} historical 1m bars "
            f"(discarded {discarded} partial 1m bars at boundary)."
        )

    def _process_new_1m_bars(self, data: pd.DataFrame) -> bool:
        new_bars_generated: bool = False
        if self.last_processed_timestamp is not None:
            # Ensure index is datetime-like before comparison
            data.index = pd.to_datetime(data.index)
            new_data = data[data.index > self.last_processed_timestamp]
        else:
            new_data = data

        if new_data.empty:
            logging.debug("No new 1m bars to process.")
            return new_bars_generated

        logging.debug(
            f"Processing {len(new_data)} new 1m bar(s) | "
            f"cumulative volume before: {self.cumulative_volume:.2f} | "
            f"threshold: {self.volume_threshold}"
        )

        for timestamp, row in new_data.iterrows():
            self.current_bar_data.append(row)
            self.cumulative_volume += float(row["volume"])

            logging.debug(
                f"  1m bar @ {timestamp} | close: {row['close']} | "
                f"volume: {row['volume']:.4f} | cumulative: {self.cumulative_volume:.2f}"
            )

            if self.cumulative_volume >= self.volume_threshold:
                self._create_volume_bar()
                new_bars_generated = True

        # Convert index to list and then access the last element
        self.last_processed_timestamp = pd.to_datetime(list(new_data.index)[-1])
        return new_bars_generated

    def _create_volume_bar(self) -> None:
        bar_df = pd.DataFrame(self.current_bar_data)

        open_price: float = float(bar_df["open"].iloc[0])
        high_price: float = float(bar_df["high"].max())
        low_price: float = float(bar_df["low"].min())
        close_price: float = float(bar_df["close"].iloc[-1])
        # Convert index to list and then access the last element
        close_time: pd.Timestamp = pd.to_datetime(list(bar_df.index)[-1])

        bar_log_returns = np.log(bar_df["close"] / bar_df["close"].shift(1)).dropna()
        intra_bar_std: float = (
            float(bar_log_returns.std()) if len(bar_log_returns) > 1 else 0.0
        )

        new_volume_bar = pd.DataFrame(
            [
                {
                    "open_price": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "close_price": close_price,
                    "total_volume": self.cumulative_volume,
                    "intra_bar_std": intra_bar_std,
                }
            ],
            index=pd.Index([close_time]),
        )

        new_volume_bar["bar_return"] = (
            new_volume_bar["close_price"] / new_volume_bar["open_price"]
        ) - 1
        bar_return: float = float(new_volume_bar["bar_return"].iloc[0])

        self.volume_bars = pd.concat([self.volume_bars, new_volume_bar])

        logging.debug(
            f"Volume bar created @ {close_time} | "
            f"open: {open_price} high: {high_price} low: {low_price} close: {close_price} | "
            f"volume: {self.cumulative_volume:.2f} | return: {bar_return:.4%} | "
            f"intra_bar_std: {intra_bar_std:.6f} | total bars: {len(self.volume_bars)}"
        )

        self.current_bar_data = []
        self.cumulative_volume = 0.0
        self.bars_since_refit += 1

    def get_signal(self, data: pd.DataFrame) -> Literal["BUY", "SELL", "HOLD"]:
        new_bars_generated: bool = self._process_new_1m_bars(data)

        if not new_bars_generated:
            logging.debug(
                "No volume bar completed this tick; skipping signal generation."
            )
            return "HOLD"

        try:
            features: pd.DataFrame = self.pipeline.step_2_feature_engineering(
                self.volume_bars.copy()
            )
            y: pd.Series = self.pipeline.step_3_labeling_and_weighting(
                self.volume_bars.copy()
            )[0]

            common_idx = features.index.intersection(y.index)
            features = features.loc[common_idx]
            y = y.loc[common_idx]

            if len(features) < self.min_bars_to_fit:
                logging.debug(
                    f"Not enough usable bars after feature engineering "
                    f"({len(features)}/{self.min_bars_to_fit}). Waiting for more data."
                )
                return "HOLD"

            if (
                not self.model_is_fit
                or self.bars_since_refit >= self.refit_every_n_bars
            ):
                logging.info(
                    f"Fitting model on {len(features)} volume bars "
                    f"(bars since last fit: {self.bars_since_refit})"
                )
                self.predictor, self.known_covariates_names = (
                    self.pipeline.fit_predictor(features, y)
                )
                self.model_is_fit = True
                self.bars_since_refit = 0
                logging.debug("Model fit complete.")

            if self.model_is_fit:
                logging.info("Model is fit, generating signal.")
                # Ensure known_covariates_names is not None
                if self.known_covariates_names is None:
                    logging.warning(
                        "known_covariates_names is None, cannot predict. Returning HOLD."
                    )
                    return "HOLD"

                predicted_class: int = self.pipeline.predict_next(
                    self.predictor, features, y, self.known_covariates_names
                )
                signal: Literal["BUY", "SELL", "HOLD"] = (
                    "BUY" if predicted_class == 1 else "SELL"
                )
                logging.debug(f"Predicted class: {predicted_class} → signal: {signal}")
                return signal

            logging.debug("Model not yet fit; no signal returned.")
            return "HOLD"

        except Exception as e:
            logging.error(f"Error generating signal: {e}", exc_info=True)
            return "HOLD"

    def get_order_size(self) -> float:
        return 1.0
