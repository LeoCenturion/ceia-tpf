# Training

Since time series violates the common assumption of independence among observations we can't randomly split the dataset into train and test. The order of the samples must be respected.
Backtesting is the process of training and testing a model respecting the order of the data.
Two main approaches are used when backtesting:
- Rolling window: e.g. train with the last 1 year of data and predict the next 1 day. Compute the error, roll the window forward and repeat the process.
- Expanding window: e.g. train with all the past data and predict the next 1 day. Compute the error, advance 1 day and repeat the process.


# Comparing investments
Usually the sharpe ratio is used $$ SR = \frac{return - baseline}{\sigma} $$ which conveys how much risk one's taking of each unit of extra return. The sharpe ratio is a statistic estimator and, therefore, subject to error which diminishes with the square root of the sample size $$ \frac{1}{\sqrt{T}} $$. See [The statistics of sharpe ratios][1].

Given two investments, the less correlated and the more different their true SR, the less time in years is needed to assert they are actually different. If the two rules are highly correlated and with similar SR several decades of data will be required.

When working with returns a useful approach is to standardize them so that the % change is expressed in relation to each asset's standard deviation (recent or historical).
The distribution of the returns of an investment will usually present one of three shapes:
- Centered Gaussian
- Positive skew: where you normally have a more pessimistic mean but periodically present with big wins.
- Negative skew: where you normally have a more optimistic mean but periodically present with big losses.
This is relevant for managing [risk](#Volatility target) and for comparing sharpe ratios since negative skews tend to have higher SR.

# Portfolio allocation

Bootstrapping: use an expanding window to apply the markowitz optimization several times. Then, for each asset, assign to it the average of all the weights obtained corresponding to that asset.
Handcrafting: Divide into groups of one, two, or three subgroups or assets. Items should be grouped according to correlation (and common sense) and then adjusted according to sharpe ratio. For more info on how to assign weights see [2]

# Forecasts

According to [2] forecasts should ideally be of continuum output and of finite range, e.g. [20, -20]. This should indicate the strength of the price movement being forecasted.
# Combining strategies

# Volatility target

# Position sizing


# References
[1]: https://www.researchgate.net/publication/228139699_The_Statistics_of_Sharpe_Ratios
[2]: [Systematic Trading - Robert Carver](https://www.systematicmoney.org/systematic-trading)
