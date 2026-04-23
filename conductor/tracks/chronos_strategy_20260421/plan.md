# Implementation Plan: Chronos Pipeline Bot Strategy

## Phase 1: Strategy Implementation

- [x] Task: Create the new strategy file at `src/app/chronos_strategy.py` 488f4dd
- [x] Task: Write failing tests for the `ChronosPalazzoStrategy`. a0fd2b6
    - [x] Create test file `tests/app/test_chronos_strategy.py`.
    - [x] Write a test to ensure the strategy class can be initialized and loads a mocked model.
    - [x] Write a test that simulates receiving market data and asserts that the strategy returns a "BUY" signal when the mocked model's prediction is positive.
    - [x] Write a test that asserts the strategy returns a "SELL" signal when the mocked model's prediction is zero or negative.
    - [x] Write a test to verify that an error is logged and no signal is returned if the model prediction fails.
- [x] Task: Implement the `ChronosPalazzoStrategy` class in `src/app/chronos_strategy.py` to pass the tests. 7030847
    - [x] Implement the `__init__` method to load the `TimeSeriesPredictor` from a path provided in the config.
    - [x] Implement the `get_signal` method.
        - [x] Add logic to rename incoming DataFrame columns to match the training pipeline's expectations (`close` -> `close_price`, etc.).
        - [x] Instantiate `PalazzoChronosBinaryClassificationPipeline` to get access to its `step_2_feature_engineering` method.
        - [x] Call the feature engineering method to transform the raw data.
        - [x] Convert the resulting feature DataFrame into the `TimeSeriesDataFrame` format required by the predictor.
        - [x] Create the `known_covariates` DataFrame for future timestamps (e.g., by forward-filling the last known values).
        - [x] Wrap the `model.predict()` call in a try/except block.
        - [x] Implement the logic to convert the numerical prediction to a "BUY" (> 0) or "SELL" (<= 0) signal.
        - [x] In the case of an exception, log the error and return `None`.
    - [x] Implement a `get_order_size` method (e.g., returning a fixed default value).
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Strategy Implementation' (Protocol in workflow.md)

## Phase 2: Integration and Configuration

- [ ] Task: Write a failing test to ensure the `StrategyFactory` can create the `ChronosPalazzoStrategy`.
- [ ] Task: Integrate the new strategy into the bot's `StrategyFactory`.
    - [ ] Modify `src/app/strategy.py` to import `ChronosPalazzoStrategy`.
    - [ ] Add a new case in the `create_strategy` method to handle `name='ChronosPalazzo'`.
- [ ] Task: Create an example configuration file.
    - [ ] Create a new file named `config.chronos.yaml.example` in the root directory.
    - [ ] Add the necessary configuration for the bot and the new strategy, including a placeholder for the `model_path`.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Integration and Configuration' (Protocol in workflow.md)
