"""
Min-Max scaling  (from your min_max.py)

Squeezes each numeric column into the range 0 to 1.
Rule: FIT the scaler on training data only, then TRANSFORM both sets.
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import config
from data_loader import load_data
from utils import banner

df = load_data()
print(df[config.SCALE_COLS].head())

train_df, test_df = train_test_split(
    df,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=df[config.TARGET],
)
train_df = train_df.copy()
test_df = test_df.copy()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

banner("Training data BEFORE scaling")
print(train_df[config.SCALE_COLS].head())

scaler = MinMaxScaler()
train_df[config.SCALE_COLS] = scaler.fit_transform(train_df[config.SCALE_COLS])
test_df[config.SCALE_COLS] = scaler.transform(test_df[config.SCALE_COLS])

print("\nOriginal dataset shape:", df.shape)
print("Scaled training shape:", train_df.shape)
print("Scaled testing shape:", test_df.shape)

banner("Scaled TRAINING data (min=0, max=1)")
print(train_df[config.SCALE_COLS].describe().loc[["min", "max"]])
print(train_df[config.SCALE_COLS].head())

banner("Scaled TESTING data (can go slightly outside 0-1)")
print(test_df[config.SCALE_COLS].head())
