import pandas as pd
from src.data_preprocessing.data_preprocessing import load_data, clean_data
from src.data_preprocessing.feature_engineering import add_technical_indicators, add_time_based_features, add_lagged_features
from src.data_preprocessing.scaling import scale_data
from src.data_preprocessing.splitting import split_data

# 1. Create some dummy data
data = {
    'timestamp': pd.to_datetime(['2023-01-01 12:00:00', '2023-01-02 13:00:00', '2023-01-03 14:00:00', '2023-01-04 15:00:00', '2023-01-05 16:00:00']),
    'open': [10, 20, 30, 40, 50],
    'high': [15, 25, 35, 45, 55],
    'low': [5, 15, 25, 35, 45],
    'close': [12, 22, 32, 42, 52],
    'volume': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

# 2. Run the preprocessing pipeline
df = clean_data(df)
df = add_time_based_features(df)
df = add_lagged_features(df, 'close', [1, 2])
df = scale_data(df)
train_df, val_df, test_df = split_data(df)

# 3. Print the results
print("Train set:")
print(train_df.head())
print("Validation set:")
print(val_df.head())
print("Test set:")
print(test_df.head())
