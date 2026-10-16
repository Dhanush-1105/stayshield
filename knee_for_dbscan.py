"""
Finding eps for DBSCAN with a k-distance graph  (from your
Knee_for_DBSCAN.py - this also replaces the empty K-distance_for_eps.py)

For every point, measure the distance to its k-th nearest neighbour,
sort them, and plot. The "knee" (where the curve bends upward) is a good
value for eps.
"""
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

from utils import finish_plot


def find_eps(X, min_samples=5):
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = neighbors.kneighbors(X)

    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(8, 5))
    plt.plot(k_distances)
    plt.xlabel("Data Points (sorted)")
    plt.ylabel(f"{min_samples}th Nearest Neighbor Distance")
    plt.title("K-Distance Graph")
    plt.grid()

    knee = KneeLocator(
        range(len(k_distances)),
        k_distances,
        curve="convex",
        direction="increasing",
        S=5,  # sensitivity: with the default (1) the knee lands at the very
              # end of the curve and eps becomes too large for DBSCAN
    )

    if knee.knee is None:
        # no clear bend: fall back to the 95th percentile of the distances
        eps = round(float(np.percentile(k_distances, 95)), 4)
    else:
        eps = round(float(k_distances[knee.knee]), 4)
        plt.axhline(eps, color="red", linestyle="--", label=f"eps = {eps}")
        plt.legend()

    finish_plot("dbscan_k_distance")
    return eps
