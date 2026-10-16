"""
K-Means with the Elbow method  (from your k-means_elbow.py)

Two features (lead_time and adr) so the clusters can be drawn on a
2-D chart. Steps:
  1. scale the two features (they are on very different scales)
  2. elbow curve for k = 1..10
  3. fit K-Means with k = 3
  4. print centroids, cluster sizes and draw the clusters
"""
import matplotlib.pyplot as plt
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- elbow method ------------------------------------------------------
inertia = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker="o", linestyle="--", color="b")
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.xticks(k_range)
plt.grid(True, linestyle="--", alpha=0.6)
finish_plot("kmeans_elbow_2d")

# ---- final model (k = 3) -----------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=config.RANDOM_STATE, n_init=10)
X["Cluster"] = kmeans.fit_predict(X_scaled)

# centroids back in original units (days / euros)
centers = scaler.inverse_transform(kmeans.cluster_centers_)

banner("Cluster Centroids")
print(pd.DataFrame(centers, columns=[col_x, col_y]))

banner("Booking Count Per Cluster")
print(X["Cluster"].value_counts().sort_index()
      .rename_axis("Cluster").reset_index(name="Count"))

plt.figure(figsize=(8, 6))
plt.scatter(X[col_x], X[col_y], c=X["Cluster"], cmap="viridis",
            alpha=0.5, edgecolors="k", linewidths=0.2, label="Bookings")
plt.scatter(centers[:, 0], centers[:, 1], s=250, c="red", marker="X",
            label="Centroids")
plt.title("Hotel Booking Clusters (K-Means)", fontsize=12)
plt.xlabel(col_x)
plt.ylabel(col_y)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
finish_plot("kmeans_clusters_2d")
