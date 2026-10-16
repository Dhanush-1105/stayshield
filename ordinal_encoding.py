"""
Ordinal encoding  (from your ordinal_encoding.py)

For columns that HAVE an order. The order is declared by us - never let
the encoder guess it (alphabetical order would be wrong here).

    arrival_date_month : January < February < ... < December
    lead_time_tier     : Short < Medium < Long

(Your original file had typos: 'category=' should be 'categories=',
'one_encoded_value' should be 'use_encoded_value', and train_df/test_df
were never created. All fixed here.)
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import config
from data_loader import load_data
from utils import banner

df = load_data()

train_df, test_df = train_test_split(
    df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
)
train_df = train_df.copy()
test_df = test_df.copy()

ord_enc = OrdinalEncoder(
    categories=config.ORDINAL_ORDERS,
    handle_unknown="use_encoded_value",   # unseen value -> -1
    unknown_value=-1,
)

ordinal_cols = config.ORDINAL_COLS
enc_cols = [c + "_enc" for c in ordinal_cols]

train_df[enc_cols] = ord_enc.fit_transform(train_df[ordinal_cols])
test_df[enc_cols] = ord_enc.transform(test_df[ordinal_cols])

banner("Declared category order")
for col, cats in zip(ordinal_cols, ord_enc.categories_):
    print(f"{col}: {list(cats)}")

banner("Training data - original vs encoded")
print(train_df[ordinal_cols + enc_cols].head(10))

banner("Testing data - original vs encoded")
print(test_df[ordinal_cols + enc_cols].head(10))

banner("Check: encoded values are 0,1,2... in the declared order")
print(train_df.groupby("lead_time_tier")["lead_time_tier_enc"].first())
print(train_df.groupby("arrival_date_month")["arrival_date_month_enc"].first()
      .sort_values())
