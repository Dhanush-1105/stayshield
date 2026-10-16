"""
Build the preprocessed dataset  (replaces your
placement_predict_50k_preprocessed.csv)

Used by the clustering scripts (K-Means and DBSCAN). Steps:
  1. load + clean + add new columns
  2. flag anomalies with the IQR rule  -> IsAnomaly (0/1)
  3. fill missing values (median for numbers, most frequent for text)
  4. standard-scale every numeric feature (clustering needs equal scales)
Text columns stay as text, exactly like your original file; the
clustering scripts only use the numeric columns.
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import config
from data_loader import get_X_y, load_data
from utils import banner

df = load_data()
print("Rows after cleaning:", len(df))

# ---- 1. anomaly flag (IQR rule on adr and lead_time) -------------------
is_anomaly = pd.Series(False, index=df.index)
for col in config.ANOMALY_COLS:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    is_anomaly |= (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
df["IsAnomaly"] = is_anomaly.astype(int)
print("Anomalies flagged:", df["IsAnomaly"].sum(),
      f"({df['IsAnomaly'].mean():.2%})")

# ---- 2. keep model columns only (no leakage) ---------------------------
X, y = get_X_y(df)
X["IsAnomaly"] = df["IsAnomaly"]
numeric_cols = X.select_dtypes(include="number").columns.tolist()
numeric_cols.remove("IsAnomaly")
text_cols = X.select_dtypes(exclude="number").columns.tolist()

# ---- 3. missing values --------------------------------------------------
banner("Missing values before filling")
print(X.isna().sum()[X.isna().sum() > 0])

X[numeric_cols] = SimpleImputer(strategy="median").fit_transform(X[numeric_cols])
X[text_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[text_cols])

# ---- 4. scale numeric columns ------------------------------------------
X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])

out = X.copy()
out[config.TARGET] = y.values

out.to_csv(config.PREPROCESSED_CSV, index=False)

banner("Done")
print("Saved:", config.PREPROCESSED_CSV)
print("Shape:", out.shape)
print("Missing values left:", int(out.isna().sum().sum()))
print("Numeric columns scaled:", len(numeric_cols))
