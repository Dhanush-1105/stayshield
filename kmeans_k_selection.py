"""
Choosing K for K-Means  (from your k_calculation_k_means.py and the
unfinished k_means_final.py)

Uses the preprocessed (scaled) numeric data. Three ways to pick K:
  1. Manual      - you type K
  2. Elbow       - look at the WCSS graph, then type K
  3. Silhouette  - the K with the best silhouette score is chosen
The chosen K is then used for the final clustering.

When run without questions (run_all.py) option 3 is used automatically,
and the elbow option picks K with the 'kneed' library.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config
from utils import banner, finish_plot, require_preprocessed

K_VALUES = range(2, 12)


def load_data():
    require_preprocessed()
    data = pd.read_csv(config.PREPROCESSED_CSV)
    X = data.drop(columns=config.CLUSTER_DROP_COLUMNS, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())
    return X, data[config.TARGET]


def manual_k():
    print("\n--- Manual K Selection ---")
    return int(input("Enter the value of k: "))


def elbow_method(X):
    wcss = []
    for k in K_VALUES:
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    print("\nWCSS Values:")
    for k, value in zip(K_VALUES, wcss):
        print("K =", k, "WCSS =", round(value, 1))

    plt.plot(K_VALUES, wcss, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.xticks(list(K_VALUES))
    finish_plot("kmeans_elbow_method")

    if config.INTERACTIVE:
        return int(input("\nEnter K based on the Elbow graph (2 to 11): "))

    knee = KneeLocator(list(K_VALUES), wcss, curve="convex", direction="decreasing")
    k = knee.knee or 3
    print("\nElbow found automatically at K =", k)
    return int(k)


def silhouette_method(X):
    scores = []
    for k in K_VALUES:
        print("Calculating silhouette score for K =", k)
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X)
        scores.append(silhouette_score(
            X, labels, sample_size=5000, random_state=config.RANDOM_STATE))

    print("\nSilhouette Scores:")
    for k, score in zip(K_VALUES, scores):
        print("K =", k, "Silhouette Score =", round(score, 4))

    best_k = list(K_VALUES)[int(np.argmax(scores))]
    print("\nBest K =", best_k)

    plt.plot(K_VALUES, scores, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method")
    plt.xticks(list(K_VALUES))
    finish_plot("kmeans_silhouette_method")
    return best_k


def main():
    banner("K-Means Clustering")
    X, canceled = load_data()
    print("Clustering data:", X.shape)

    if config.INTERACTIVE:
        print("\n1. Manual K Selection")
        print("2. Elbow Method")
        print("3. Silhouette Method")
        choice = int(input("\nEnter your choice: "))
    else:
        choice = 3

    if choice == 1:
        k = manual_k()
    elif choice == 2:
        k = elbow_method(X)
    elif choice == 3:
        k = silhouette_method(X)
    else:
        print("Invalid choice")
        return

    if k < 2 or k > 11:
        print("K must be between 2 and 11")
        return

    # ---- final clustering ----------------------------------------------
    kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)

    banner(f"Final K-Means with K = {k}")
    result = pd.DataFrame({"Cluster": labels, "is_canceled": canceled.values})
    summary = (result.groupby("Cluster")["is_canceled"]
               .agg(Bookings="size", Cancellation_rate="mean").round(3))
    print(summary)

    out = config.OUTPUT_DIR / "kmeans_final_clusters.csv"
    result.to_csv(out, index=False)
    print("\nSaved:", out.name)


if __name__ == "__main__":
    main()
