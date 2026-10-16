"""
Run the WHOLE StayShield project with one click.

In PyCharm: right-click this file -> Run 'run_all'.
Plots are saved to the outputs/ folder instead of popping up, and no
questions are asked, so it runs from start to finish by itself
(about 2-3 minutes).

To run a single algorithm with pop-up plots, open that file
(for example adaboost.py) and run it directly.
"""
import os
import subprocess
import sys
import time

import config

STEPS = [
    ("Preprocessing (builds the clustering dataset)", "preprocessing.py"),
    ("Min-Max scaling",                               "min_max.py"),
    ("Standard scaling",                              "standard.py"),
    ("Outlier handling (IQR)",                        "outlier.py"),
    ("One-Hot encoding",                              "one_hot_encoding.py"),
    ("Ordinal encoding",                              "ordinal_encoding.py"),
    ("Linear Regression",                             "linear_regression.py"),
    ("Logistic Regression",                           "logistic_regression.py"),
    ("AdaBoost",                                      "adaboost.py"),
    ("Gradient Boosting",                             "gradient_boosting.py"),
    ("K-Means (elbow, 2-D)",                          "kmeans_elbow.py"),
    ("K-Means (clusters + outliers)",                 "kmeans_outliers.py"),
    ("K-Means (choosing K)",                          "kmeans_k_selection.py"),
    ("DBSCAN",                                        "dbscan.py"),
]


def main():
    env = os.environ.copy()
    env["STAYSHIELD_SHOW_PLOTS"] = "0"
    env["STAYSHIELD_INTERACTIVE"] = "0"

    results = []
    for name, script in STEPS:
        print("\n" + "#" * 70)
        print(f"# {name}   ({script})")
        print("#" * 70)
        start = time.time()
        code = subprocess.run(
            [sys.executable, str(config.PROJECT_ROOT / script)],
            cwd=config.PROJECT_ROOT, env=env,
        ).returncode
        results.append((name, "OK" if code == 0 else "FAILED",
                        time.time() - start))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, status, seconds in results:
        print(f"{status:7s} {seconds:6.1f}s  {name}")
    print(f"\nPlots and result files are in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
