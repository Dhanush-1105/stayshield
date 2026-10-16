"""
Standard scaling  (from your standard.py)

Rescales each column to mean = 0 and standard deviation = 1.
(Your original called fit_transform twice on the training data; here it
is fitted exactly once.)
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()
train_df[config.SCALE_COLS] = scaler.fit_transform(train_df[config.SCALE_COLS])
test_df[config.SCALE_COLS] = scaler.transform(test_df[config.SCALE_COLS])

print("\nOriginal dataset shape:", df.shape)
print("Scaled training shape:", train_df.shape)
print("Scaled testing shape:", test_df.shape)

banner("Scaled TRAINING data (mean ~ 0, std ~ 1)")
print(train_df[config.SCALE_COLS].agg(["mean", "std"]).round(3))
print(train_df[config.SCALE_COLS].head())

banner("Scaled TESTING data")
print(test_df[config.SCALE_COLS].head())
