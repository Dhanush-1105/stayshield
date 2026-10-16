"""
DBSCAN  (from your DBSCAN.py)

Density-based clustering. It finds:
  core points   - have at least min_samples neighbours within eps
  border points - inside a cluster but with fewer neighbours
  noise points  - belong to no cluster (label -1)
eps is chosen automatically with the k-distance knee.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import config
from knee_for_dbscan import find_eps
from utils import banner, finish_plot, require_preprocessed

require_preprocessed()
data = pd.read_csv(config.PREPROCESSED_CSV)

# 119,000 rows is slow for DBSCAN -> use a random sample (same as your original)
if len(data) > config.DBSCAN_SAMPLE_SIZE:
    data = data.sample(config.DBSCAN_SAMPLE_SIZE, random_state=config.RANDOM_STATE)

X = data.drop(columns=config.CLUSTER_DROP_COLUMNS, errors="ignore")
X = X.select_dtypes(include=[np.number])
X = X.fillna(X.median())

eps = find_eps(X, min_samples=config.DBSCAN_MIN_SAMPLES)
print("EPS:", eps)

model = DBSCAN(eps=eps, min_samples=config.DBSCAN_MIN_SAMPLES)
labels = model.fit_predict(X)

core = model.core_sample_indices_
noise = labels == -1
core_mask = pd.Series(range(len(X))).isin(core).to_numpy()
border = ~noise & ~core_mask

banner("DBSCAN RESULTS")
print("Clusters:", len(set(labels)) - (1 if -1 in labels else 0))
print("Core:", len(core))
print("Border:", border.sum())
print("Noise:", noise.sum())

# Do the noise points cancel more often? (extra insight for StayShield)
canceled = data[config.TARGET].to_numpy()
if noise.any():
    print(f"\nCancellation rate - noise points : {canceled[noise].mean():.1%}")
    print(f"Cancellation rate - clustered    : {canceled[~noise].mean():.1%}")

plt.figure(figsize=(8, 6))
clustered = ~noise
plt.scatter(X.loc[clustered, config.PLOT_X], X.loc[clustered, config.PLOT_Y],
            c=labels[clustered], cmap="tab10", s=15, label="Clustered points")
plt.scatter(X.loc[noise, config.PLOT_X], X.loc[noise, config.PLOT_Y],
            c="red", marker="x", s=45, label="Noise (-1)")
plt.xlabel(config.PLOT_X + " (scaled)")
plt.ylabel(config.PLOT_Y + " (scaled)")
plt.title("DBSCAN Clustering (2-D view of a 20-feature result)")
plt.legend()
finish_plot("dbscan_clusters")
