#%% [markdown]
# BTC Price Data Analysis (Percentage Change)
#
# This script performs a basic analysis of BTC/USDT hourly price data, focusing on the behavior of the percentage change in the close price.

#%% [markdown]
# ## 1. Load Data and Libraries

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# Set plot style
sns.set(style="whitegrid")

print("Libraries loaded.")

#%% [markdown]
# ### Helper Functions

#%%
def sma(series, n):
    """Calculates the Simple Moving Average."""
    return series.rolling(window=n).mean()

def awesome_oscillator(high: pd.Series, low: pd.Series, fast_period: int = 5, slow_period: int = 34) -> pd.Series:
    """Calculates the Awesome Oscillator."""
    median_price = (high + low) / 2
    ao = sma(median_price, fast_period) - sma(median_price, slow_period)
    return ao

#%% [markdown]
# ## 2. Load and Prepare Data

#%%
DATA_PATH = 'data/BTCUSDT_1h.csv'
df = pd.read_csv(DATA_PATH)

# Convert to datetime and set as index
df['timestamp'] = pd.to_datetime(df['date'])
df.set_index('timestamp', inplace=True)
df.sort_index(inplace=True)

print("Data loaded and prepared. Shape:", df.shape)
print(df.head())

#%% [markdown]
# ## 3. Analyze Close Price Percentage Change
# 
# We calculate the percentage change of the closing price. This normalizes the price changes and is a standard way to analyze financial time series. It's also a crucial step for time series modeling, as models like ARIMA often require stationary data.

#%%
# Calculate the percentage change in the closing price
df['close_pct_change'] = (df['close'].pct_change()) * 100

# Drop the first row with a NaN value resulting from the pct_change operation
df.dropna(subset=['close_pct_change'], inplace=True)

print("Close price percentage change calculated.")
print(df[['close', 'close_pct_change']].head())

#%% [markdown]
# ### 3.1. Time Series Plot of Price Percentage Change

#%%
print("Plotting the close price percentage change over time...")
plt.figure(figsize=(15, 7))
plt.plot(df.index, df['close_pct_change'], label='Close Price Percentage Change', linewidth=0.7)
plt.title('BTC/USDT Hourly Close Price Percentage Change Over Time')
plt.xlabel('Date')
plt.ylabel('Price Change (%)')
plt.legend()
plt.show()

#%% [markdown]
# The plot shows that the percentage change is centered around zero but exhibits volatility clustering—periods of high volatility are followed by more high volatility, and vice versa.

#%% [markdown]
# ### 3.2. Distribution of Price Percentage Change

#%%
print("Plotting the distribution of the close price percentage change...")
plt.figure(figsize=(12, 6))
sns.histplot(df['close_pct_change'], bins=100, kde=True)
plt.title('Distribution of Hourly Close Price Percentage Change')
plt.xlabel('Price Change (%)')
plt.ylabel('Frequency')
plt.show()

#%% [markdown]
# The distribution is highly leptokurtic (sharply peaked at zero with heavy tails), which is a classic characteristic of financial returns. This indicates that extreme price changes occur more frequently than a normal distribution would predict.

#%% [markdown]
# ### 3.3. Descriptive Statistics

#%%
print("Descriptive Statistics for Close Price Percentage Change:")
print(df['close_pct_change'].describe())

#%% [markdown]
# The mean is very close to zero. The standard deviation confirms the significant volatility, and the min/max values highlight the extreme hourly percentage swings present in the data.

#%% [markdown]
# ### 3.4. Stationarity Test (Augmented Dickey-Fuller)
# 
# We perform an ADF test to formally check if the percentage change series is stationary. The null hypothesis is that the series is non-stationary.

#%%
print("Performing Augmented Dickey-Fuller (ADF) test for stationarity...")
adf_result = adfuller(df['close_pct_change'])

print(f'ADF Statistic: {adf_result[0]:.4f}')
print(f'p-value: {adf_result[1]:.4f}')
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'\t{key}: {value:.3f}')

if adf_result[1] <= 0.05:
    print("\nResult: The series is stationary (p-value <= 0.05).")
else:
    print("\nResult: The series is not stationary (p-value > 0.05).")


#%% [markdown]
# The extremely low p-value allows us to confidently reject the null hypothesis, confirming that the percentage change series is stationary. This is a suitable input for time series models.

#%% [markdown]
# ### 3.5. Zoomed-in Histogram of Price Percentage Change

#%%
print("Plotting the zoomed-in distribution for price changes between -0.1% and 0.1%...")
zoomed_data = df['close_pct_change'].loc[(df['close_pct_change'] >= -0.1) & (df['close_pct_change'] <= 0.1)]

plt.figure(figsize=(12, 6))
sns.histplot(zoomed_data, bins=50, kde=True)
plt.title('Zoomed-in Distribution of Hourly Close Price Percentage Change (between -0.1% and 0.1%)')
plt.xlabel('Price Change (%)')
plt.ylabel('Frequency')
plt.show()

#%% [markdown]
# Zooming in on the vast majority of the data, we can see the distribution around zero more clearly.

#%% [markdown]
# ### 3.6. Cumulative Distribution of Absolute Price Change

#%%
print("Plotting the Cumulative Distribution Function (CDF) of absolute percentage changes...")
df['abs_close_pct_change'] = df['close_pct_change'].abs()

plt.figure(figsize=(12, 6))
sns.ecdfplot(df['abs_close_pct_change'])
plt.title('CDF of Absolute Hourly Percentage Price Change')
plt.xlabel('Absolute Price Change (%)')
plt.ylabel('Cumulative Probability')
plt.grid(True, which="both", ls="--")
plt.show()

#%% [markdown]
# The CDF plot shows the probability of observing a price change of a certain magnitude or less. For example, you can use it to find the value on the x-axis that corresponds to 0.95 on the y-axis to see what magnitude of price change accounts for 95% of all hourly movements.

#%% [markdown]
# ## 4. Awesome Oscillator Analysis

#%% [markdown]
# We will now analyze the Awesome Oscillator (AO). We'll compute it in two ways:
# 1. Using the raw `High` and `Low` prices.
# 2. Using the percentage change of the `High` and `Low` prices.
# 
# This comparison will help visualize how the oscillator behaves on the price series versus the returns series.

#%%
# Calculate AO on raw prices
print("Calculating Awesome Oscillator on raw prices...")
df['ao_price'] = awesome_oscillator(df['high'], df['low'])

# Calculate AO on percentage change
print("Calculating Awesome Oscillator on price percentage changes...")
df['high_pct_change'] = df['high'].pct_change()
df['low_pct_change'] = df['low'].pct_change()

# Fill NaNs created by pct_change for AO calculation
high_pct_change_filled = df['high_pct_change'].fillna(0)
low_pct_change_filled = df['low_pct_change'].fillna(0)
df['ao_pct_change'] = awesome_oscillator(high_pct_change_filled, low_pct_change_filled)

print("Awesome Oscillators calculated.")
print(df[['ao_price', 'ao_pct_change']].head())

#%% [markdown]
# ### 4.1. Plotting Awesome Oscillators

#%%
print("Plotting Awesome Oscillators for comparison...")
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot AO on price
axes[0].plot(df.index, df['ao_price'], label='AO on Price', color='blue', linewidth=0.8)
axes[0].set_title('Awesome Oscillator on Raw Price (High/Low)')
axes[0].set_ylabel('AO Value')
axes[0].grid(True)
axes[0].legend()

# Plot AO on percentage change
axes[1].plot(df.index, df['ao_pct_change'], label='AO on % Change', color='green', linewidth=0.8)
axes[1].set_title('Awesome Oscillator on Price Percentage Change')
axes[1].set_ylabel('AO Value')
axes[1].grid(True)
axes[1].legend()

plt.xlabel('Date')
fig.tight_layout()
plt.show()

#%% [markdown]
# The plots show a stark difference. The AO on raw prices reflects longer-term price trends and momentum, with large swings. The AO on percentage change is much more stationary and centered around zero, reflecting the short-term volatility and returns behavior rather than the price level itself.

#%% [markdown]
# ## 5. Consecutive Runs Analysis
#
# Here, we analyze the length of consecutive periods of positive or negative price changes. A "run" is a sequence of one or more hours where the price change has the same sign. For example:
# - A single hour of gains is a run of `+1`.
# - Three consecutive hours of gains is a run of `+3`.
# - Two consecutive hours of losses is a run of `-2`.
#
# This analysis can help identify if there is momentum or mean-reversion behavior in the short term.

#%%
print("Analyzing consecutive runs of positive and negative price changes...")

# Determine the sign of the price change, treating 0 as neutral
signs = np.sign(df['close_pct_change'])
# We replace 0s with the previous sign to not break a run
signs = signs.replace(0, np.nan).ffill().fillna(0)

# Identify blocks of consecutive signs
blocks = signs.diff().ne(0).cumsum()

# Calculate the length of each run and multiply by its sign
runs = signs.groupby(blocks).apply(lambda x: x.size * x.iloc[0] if x.iloc[0] != 0 else 0)

# Filter out any zero-runs that might have occurred at the beginning
runs = runs[runs != 0]

print("Consecutive runs calculated. Descriptive statistics:")
print(runs.describe())

#%%
# Plot a histogram of the runs
print("\nPlotting the histogram of consecutive runs...")
plt.figure(figsize=(14, 7))
sns.histplot(runs, discrete=True, stat="count", shrink=0.8)
plt.title('Histogram of Consecutive Hourly Price Change Runs')
plt.xlabel('Run Length (Negative = Losses, Positive = Gains)')
plt.ylabel('Frequency (Count of Runs)')
plt.xticks(np.arange(int(runs.min()), int(runs.max()) + 1, 1))
plt.grid(True, which="both", ls="--")
plt.show()

#%% [markdown]
# The histogram shows that short runs (of length 1, 2, or 3) are very common, while long streaks of consecutive gains or losses are increasingly rare. This is consistent with the behavior of a volatile asset where the direction can change frequently.

#%% [markdown]
# ### 5.1. Run Transition Probability Analysis
#
# Now, we'll analyze the probability of transitioning from a run of one length to another. For example, what is the likelihood that a 1-hour run of gains is followed by a 1-hour run of losses? This can be modeled as a Markov chain, and we can compute the transition matrix.
#
# We will focus on transitions between runs of length -5 to +5.

#%%
print("Calculating run transition probabilities...")
# AI compute the sharp ratio and other indicators from the whole data AI!
# Create a DataFrame of consecutive run pairs
transitions = pd.DataFrame({'from': runs, 'to': runs.shift(-1)}).dropna()
transitions = transitions.astype(int)

# Create a transition count matrix using crosstab
transition_counts = pd.crosstab(transitions['from'], transitions['to'])

# Calculate transition probabilities by dividing each row by its sum
transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0)

# Filter and reindex to focus on the -5 to +5 range, filling missing transitions with 0
run_range = range(-5, 6)
# Remove 0 from the range as it's not a valid run length in our data
run_range = [i for i in run_range if i != 0]

transition_matrix = transition_probabilities.reindex(index=run_range, columns=run_range, fill_value=0)

print("Transition Matrix (-5 to +5):")
print(transition_matrix.to_string(float_format="%.3f"))

#%%
# Plot the transition matrix as a heatmap
print("\nPlotting the transition probability heatmap...")
plt.figure(figsize=(12, 10))
sns.heatmap(transition_matrix.T, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
plt.title('Transition Probability Between Consecutive Run Lengths (-5 to +5)')
plt.xlabel('From Run Length')
plt.ylabel('To Run Length')
plt.show()

#%% [markdown]
# The heatmap shows the probability of moving from a run length on the x-axis to a run length on the y-axis.
#
# A key observation is the high probability along the anti-diagonal, especially for short runs. For example, there's a high likelihood that a run of `+1` is followed by a run of `-1`, and vice-versa. This suggests a strong tendency for the price direction to revert after short periods, which aligns with the previous observation that long runs are rare.

#%% [markdown]
# ### 5.2. Conditional Probability of Run Continuation
#
# Here we analyze the probability that a run of a certain length will continue for at least one more period. For example, given that we have observed two consecutive hours of price increases (a run of +2), what is the probability that the next hour will also be an increase, extending the run to +3?
#
# This is calculated for positive and negative runs separately. The probability is:
# `P(Run extends to N+1 | Run has reached length N) = (Total number of runs with length >= N+1) / (Total number of runs with length >= N)`

#%%
print("Calculating conditional probabilities of run continuation...")

positive_runs = runs[runs > 0]
negative_runs = runs[runs < 0]

continuation_probs = {
    'Positive': [],
    'Negative': []
}
run_lengths_to_check = range(1, 6)

for n in run_lengths_to_check:
    # Positive runs
    runs_ge_n = (positive_runs >= n).sum()
    runs_ge_n_plus_1 = (positive_runs >= n + 1).sum()
    prob_pos = runs_ge_n_plus_1 / runs_ge_n if runs_ge_n > 0 else 0
    continuation_probs['Positive'].append(prob_pos)

    # Negative runs
    runs_le_n = (negative_runs <= -n).sum()
    runs_le_n_plus_1 = (negative_runs <= -(n + 1)).sum()
    prob_neg = runs_le_n_plus_1 / runs_le_n if runs_le_n > 0 else 0
    continuation_probs['Negative'].append(prob_neg)

prob_df = pd.DataFrame(continuation_probs, index=[f'{i} -> {i+1}' for i in run_lengths_to_check])
prob_df.index.name = 'Run Extension'

print("Conditional Probabilities of Run Continuation:")
print(prob_df.to_string(float_format="%.3f"))

#%%
# Plot the continuation probabilities
print("\nPlotting the run continuation probabilities...")
prob_df.plot(kind='bar', figsize=(14, 7), rot=0)
plt.title('Conditional Probability of a Run Continuing')
plt.xlabel('Run Length Extension (From N to N+1)')
plt.ylabel('Probability')
plt.grid(axis='y', linestyle='--')
plt.legend(title='Run Type')
plt.show()

#%% [markdown]
# The bar chart shows the probability that a run of length N will extend to N+1. For both positive and negative runs, the probability of continuation is consistently below 0.5 and generally decreases as the run gets longer. This reinforces the idea of mean reversion in the hourly price changes; the longer a directional streak continues, the more likely it is to break.
