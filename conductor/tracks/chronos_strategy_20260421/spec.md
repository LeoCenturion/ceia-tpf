# Specification: Chronos Pipeline Bot Strategy

## 1. Overview

This track involves implementing the pre-trained `PalazzoChronosBinaryClassificationPipeline` as a new, live trading strategy for the bot. The strategy will load a model trained by the existing pipeline, generate features from live market data, and produce "BUY" or "SELL" signals.

## 2. Functional Requirements

- **Model Loading:** The strategy must load a pre-trained Chronos model.
  - The path to the model directory will be specified in a configuration file (e.g., `config.yaml`).
- **Signal Generation:** The strategy must be able to generate a "BUY" or "SELL" signal from incoming k-line data.
  - It will reuse the exact feature engineering logic from `PalazzoChronosBinaryClassificationPipeline` to ensure consistency.
  - It will format the data into the `TimeSeriesDataFrame` structure required by the model.
- **Prediction Logic:**
  - The strategy will use the loaded model to predict future price movement.
  - A model output > 0 will be interpreted as a "BUY" signal.
  - A model output <= 0 will be interpreted as a "SELL" signal.
- **Data Handling:**
  - For each prediction, the strategy will fetch the most recent 512 bars of historical data to use as context, matching the `context_length` used during model training.
- **Error Handling:**
  - If the model fails to produce a prediction for any reason, the failure will be logged, and the bot will hold its current position without taking any action.

## 3. Non-Functional Requirements

- **Configurability:** The model path must be configurable and not hardcoded.
- **Performance:** Feature generation and prediction should be completed efficiently to not miss trading opportunities.

## 4. Acceptance Criteria

- A new strategy named `ChronosPalazzoStrategy` is available in the `StrategyFactory`.
- When configured with a valid model path, the bot can successfully load the model and start.
- The strategy correctly generates "BUY" and "SELL" signals based on live data and model predictions.
- In case of a model prediction error, the error is logged, and no trade is executed.

## 5. Out of Scope

- Training the Chronos model. This track only covers the implementation of a *pre-trained* model.
- Real-time model updates or re-training.
- Live trading execution logic (this is handled by the bot's core framework).
