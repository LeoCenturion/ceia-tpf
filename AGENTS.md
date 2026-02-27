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
This project uses poetry. Dependencies are defined in pyproject.toml. To run python you should ALWAYS use poetry e.g. 'poetry run python -m src.modeling.my_model'

Utility scripts are in scripts/

Data analysis notebooks and sctipts are found in data_analysis/. Before modifying files there you should always run make sync-notebooks. After creating a new script/notebooks you should run make pair-notebooks
