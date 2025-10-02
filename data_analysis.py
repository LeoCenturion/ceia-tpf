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