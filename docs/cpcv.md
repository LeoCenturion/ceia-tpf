Combinatorial Purged Cross-Validation (CPCV) is an advanced backtesting methodology designed by Marcos López de Prado to address the limitations of standard historical simulations (Walk-Forward) and standard cross-validation. Instead of generating a single historical path, CPCV systematically tests multiple scenarios to create a distribution of performance outcomes and prevent data leakage.
The logic of CPCV follows these sequential steps:
Step 1: Data Partitioning Partition the total time series of T observations into N chronological groups without shuffling
. The first N−1 groups are of equal size ⌊T/N⌋, and the final Nth group absorbs the remaining observations
.
Step 2: Combinatorial Splitting Compute all possible combinations for training and testing splits. If you decide the testing set will consist of k groups, the remaining N−k groups will constitute the training set
. This combinatorial math generates (kN​) unique splits
.
Step 3: Purging and Embargoing (Leakage Prevention) For every training/testing split, you must remove any overlapping information to prevent data leakage:
• Purging: Delete any observations from the training set whose label spans over a time period that was used to determine the label of a testing set observation
.
• Embargoing: Because financial data exhibits long memory and serial correlation, apply an embargo by deleting a small buffer of training observations that occur immediately after the testing set
.
Step 4: Model Training and Forecasting Fit your machine learning classifiers on each of the (N−kN​) purged and embargoed training sets
. Once trained, use the models to produce out-of-sample forecasts for their respective k-group testing sets
.
Step 5: Backtest Path Construction Assemble the forecasts from the tested groups to construct multiple complete backtest paths
. The combinatorial structure ensures that the tested groups are uniformly distributed, allowing you to generate ϕ[N,k] distinct backtest paths out of the same historical data
.
Step 6: Performance Distribution and PBO Evaluation Unlike traditional backtests that yield a single Sharpe ratio, calculate the performance metric (e.g., Sharpe ratio) for each of the generated backtest paths
. This results in an empirical distribution of Sharpe ratios. Finally, use this distribution to calculate the Probability of Backtest Overfitting (PBO), which tells you the statistical probability that the strategy's apparent success is simply the result of selection bias on a specific historical path.
