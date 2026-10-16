"""
K-Means clusters + outliers + cancellation profile
(from your k-means_outliers.py)

Fits K-Means (k = 3) on lead_time and adr, then:
  * profiles each cluster (size, cancellation rate)
  * marks the 1% of bookings that are farthest from their cluster
    centre as outliers
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config
from data_loader import load_data
from utils import banner, finish_plot

data = load_data()
data.columns = data.columns.str.strip()

col_x, col_y = config.CLUSTER_2D_FEATURES
print(f"Selected columns: '{col_x}' and '{col_y}'")

X = data[[col_x, col_y]].copy().dropna()
canceled = data.loc[X.index, config.TARGET]

print("\nFeature Head:")
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=config.RANDOM_STATE, n_init=10)
X["Cluster"] = kmeans.fit_predict(X_scaled)

centers = scaler.inverse_transform(kmeans.cluster_centers_)

banner("Cluster Centroids")
print(pd.DataFrame(centers, columns=[col_x, col_y]))

# ---- cluster profile ---------------------------------------------------
profile = (
    X.assign(is_canceled=canceled)
    .groupby("Cluster")
    .agg(Count=("Cluster", "size"),
         Avg_lead_time=(col_x, "mean"),
         Avg_adr=(col_y, "mean"),
         Cancellation_rate=("is_canceled", "mean"))
    .round(3)
)
banner("Cluster Profile")
print(profile)

# ---- outliers: farthest 1% from own centroid ---------------------------
dist = np.linalg.norm(
    X_scaled - kmeans.cluster_centers_[X["Cluster"].to_numpy()], axis=1
)
threshold = np.percentile(dist, 99)
X["Outlier"] = dist > threshold

banner("Outliers (farthest 1% from their cluster centre)")
print("Distance threshold:", round(threshold, 3))
print("Number of outliers:", int(X["Outlier"].sum()))
print(X[X["Outlier"]].sort_values(col_y, ascending=False).head(10))

# ---- plot --------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X[col_x], X[col_y], c=X["Cluster"], cmap="viridis",
            alpha=0.4, edgecolors="none", label="Bookings")
plt.scatter(X.loc[X["Outlier"], col_x], X.loc[X["Outlier"], col_y],
            facecolors="none", edgecolors="red", s=40, label="Outliers")
plt.scatter(centers[:, 0], centers[:, 1], s=250, c="black", marker="X",
            label="Centroids")
plt.title("K-Means Clusters and Outliers", fontsize=12)
plt.xlabel(col_x)
plt.ylabel(col_y)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
finish_plot("kmeans_outliers")
