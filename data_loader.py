"""
Loading and cleaning the hotel bookings data.

Every script calls load_data() so they all start from identical,
cleaned data. No scaling or encoding happens here - those steps belong
AFTER the train/test split so the test set never leaks into training.
"""
import numpy as np
import pandas as pd

import config


def load_raw():
    """Read the original CSV exactly as it is."""
    return pd.read_csv(config.RAW_CSV)


def clean_data(df):
    """Remove impossible rows."""
    df = df.copy()

    # Bookings with zero guests are data-entry mistakes
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    df = df[guests > 0]

    # Negative price (1 row) and the 5400 EUR price (1 row) are errors
    df = df[(df["adr"] >= 0) & (df["adr"] <= 5000)]

    return df.reset_index(drop=True)


def add_features(df):
    """Create new columns from existing ones (row-by-row, no leakage)."""
    df = df.copy()

    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"].fillna(0) + df["babies"]

    # agent is an ID with 300+ values and 14% missing -> just keep "has agent"
    df["has_agent"] = df["agent"].notna().astype(int)

    # Short / Medium / Long booking window (this is the ordinal column)
    df["lead_time_tier"] = pd.cut(
        df["lead_time"],
        bins=[-1, 30, 120, np.inf],
        labels=config.LEAD_TIME_ORDER,
    ).astype(str)

    # country has 177 values -> keep the 15 biggest, group the rest
    top_countries = df["country"].value_counts().head(15).index
    df["country_group"] = np.where(
        df["country"].isna(), np.nan,
        np.where(df["country"].isin(top_countries), df["country"], "Other"),
    )
    return df


def load_data():
    """Raw CSV -> cleaned data with new features (still has every column)."""
    return add_features(clean_data(load_raw()))


def get_X_y(df=None):
    """
    Split the data into model inputs (X) and the target (y).
    Removes the leakage columns and the columns we replaced.
    """
    if df is None:
        df = load_data()
    drop = [config.TARGET] + config.LEAKAGE_COLUMNS + config.REPLACED_COLUMNS
    X = df.drop(columns=drop, errors="ignore")
    y = df[config.TARGET]
    return X, y


def split_feature_types(X):
    """Return (numeric_columns, categorical_columns) of X."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical
