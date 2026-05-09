# Financial time series prediction using transformers.

This project focuses on predicting the price of BTC using transformer architecture.
It uses as baselines various statistical and machine learning models

It follows the machine learning cycle outlined by Lopez de Prado that consists of

1. Data Curators: collecting, cleaning, indexing, storing, adjusting, and delivering all data to the rest of the production chain.

2. Feature Analysts:raw data is transformed into informative signals or features that have predictive power. Their sole purpose is to discover, collect, and catalog libraries of predictive features (e.g., identifying patterns in order cancellations or entropy) that can be utilized by multiple other stations across the firm

3. Modeling: review the cataloged feature libraries to develop actual investment algorithms
. Their job is to make sense of the features and formulate a general economic theory or hypothesis that explains why the pattern exists (e.g., identifying a behavioral bias or structural break).

4. Backtesters Once a strategy prototype is complete, backtesters rigorously assess its profitability under various scenarios. Rather than just running a historical simulation, they evaluate the strategy against alternative scenarios and calculate the Probability of Backtest Overfitting (PBO). To prevent selection bias and overfitting, backtesters report their results directly to management, not back to the strategists.

In the `src` directory you'll find

- data_analysis: corresponding to data curation and feature engineering
- modeling: with several strategies and the utilities to run them (K Fold purged CV for example)
- backtesting: with several backtesting strategies and utilities to run such backtests

This project uses mlflow for logging results. The db is located in `mlflow.db`
This project uses optuna for hyperparameter search. The db is located in `optuna-study.db`

This project uses python. Try to document main functions. Try to use type hints whenever possible.

## Performance

Signal generators in `src/modeling/trading/` operate on datasets of 3M+ rows (1-minute BTCUSDT bars). **Vectorized pandas/numpy operations must be prioritized over Python loops.** The ffill pattern handles latching state machines without loops:

```python
sig = pd.Series(np.nan, index=index)
sig[buy_condition]  = 1
sig[sell_condition] = 0
sig.ffill().fillna(0).astype(int)
```

Python loops over large time series (even "simple" ones) have caused single trials to take 1.8 hours. Any new signal generator must be vectorized by default.
This project uses poetry. Dependencies are defined in pyproject.toml. To run python you should ALWAYS use poetry e.g. 'poetry run python -m src.modeling.my_model'

Utility scripts are in scripts/

Data analysis notebooks and sctipts are found in data_analysis/. Before modifying files there you should always run make sync-notebooks. After creating a new script/notebooks you should run make pair-notebooks

## Dependency Rules

- No module in `src/modeling` can import from `src/backtesting`

# Trading bot

in `src/app` you'll find a configurable trading bot. It pulls data from Binance exchange or from a mock exchange, feeds it to a given strategy to obtain signals and then buys or sells acordingly.
