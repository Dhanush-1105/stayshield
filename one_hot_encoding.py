"""
One-Hot encoding  (from your one_hot_encoding.py)

For columns with NO natural order (hotel type, meal, market segment...).
Each category becomes its own 0/1 column.
drop="first" removes one column per feature to avoid redundancy.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import config
from data_loader import load_data
from utils import banner

df = load_data()
print("Dataset shape:", df.shape)

nominal_cols = config.NOMINAL_COLS
print("\nNominal columns:", nominal_cols)

# country_group has a few missing values -> fill so the encoder can handle them
df[nominal_cols] = df[nominal_cols].fillna("Unknown")

train_df, test_df = train_test_split(
    df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
)
print("\nTraining data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

ohe = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore",   # unseen categories in test data -> all zeros
)

train_ohe = ohe.fit_transform(train_df[nominal_cols])
test_ohe = ohe.transform(test_df[nominal_cols])
ohe_cols = ohe.get_feature_names_out(nominal_cols)

train_ohe_df = pd.DataFrame(train_ohe, columns=ohe_cols, index=train_df.index)
test_ohe_df = pd.DataFrame(test_ohe, columns=ohe_cols, index=test_df.index)

print("\nNumber of original categorical columns:", len(nominal_cols))
print("Number of generated encoded columns:", len(ohe_cols))

banner("Generated One-Hot columns")
for col in ohe_cols:
    print(col)

banner("First 5 rows of encoded TRAINING data")
print(train_ohe_df.head())

banner("First 5 rows of encoded TEST data")
print(test_ohe_df.head())
