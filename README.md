# StayShield - Hotel Booking Cancellation Project

Predicts which hotel bookings will be **cancelled** (`is_canceled`) and
also predicts the nightly price (`adr`). Built from your "regular"
project, with every algorithm adapted to the `hotel_bookings.csv` data.

## How to run in PyCharm

1. Open this folder as a project (File > Open).
2. Create/select a Python interpreter, then install the packages:
   `pip install -r requirements.txt`
3. **Run everything:** right-click `run_all.py` > Run.
   Plots are saved in `outputs/`.
4. **Run one algorithm:** open its file and click Run. Plots pop up
   and (where it makes sense) it asks you questions.

## Files (your original file -> new file)

| Algorithm | Original | New |
|---|---|---|
| Min-Max scaling | min_max.py | `min_max.py` |
| Standard scaling | standard.py | `standard.py` |
| Outliers (IQR) | outlier.py | `outlier.py` |
| One-Hot encoding | one_hot_encoding.py | `one_hot_encoding.py` |
| Ordinal encoding | ordinal_encoding.py | `ordinal_encoding.py` |
| Preprocessed dataset | placement_..._preprocessed.csv | `preprocessing.py` |
| Linear Regression | Linear_Regression_Sklearn.py | `linear_regression.py` |
| Logistic Regression (2 scalers) | Logistic_regression_scalar.py | `logistic_regression.py` |
| AdaBoost | ADAboost.py | `adaboost.py` |
| Gradient Boosting | gradient_boosting.py | `gradient_boosting.py` |
| K-Means elbow | k-means_elbow.py | `kmeans_elbow.py` |
| K-Means outliers | k-means_outliers.py | `kmeans_outliers.py` |
| Choosing K (manual / elbow / silhouette) | k_calculation_k_means.py, k_means_final.py | `kmeans_k_selection.py` |
| DBSCAN + k-distance knee | DBSCAN.py, Knee_for_DBSCAN.py | `dbscan.py`, `knee_for_dbscan.py` |

Shared code: `config.py` (all settings and column lists),
`data_loader.py` (loading, cleaning, new columns), `utils.py` (plots).

## What changed because of the new data

| Original (placement) | StayShield (hotel) |
|---|---|
| Target `PlacementStatus` | `is_canceled` |
| Regression target `Salary Package` | `adr` (price per night) |
| `CollegeTier`, `CGPA_Tier` (ordinal) | `arrival_date_month`, `lead_time_tier` (ordinal) |
| Gender, City, Stream... (nominal) | hotel, meal, market_segment, deposit_type... |
| Removed `StudentID`, `Salary Package` | Removed `reservation_status`, `reservation_status_date` |

**Important:** `reservation_status` contains the word "Canceled", i.e. the
answer itself. Keeping it gives a fake ~100% accuracy, so it is removed
(see `LEAKAGE_COLUMNS` in `config.py`).

Cleaning and new columns (in `data_loader.py`): removed bookings with zero
guests, negative price and one 5,400 EUR price; added `total_nights`,
`total_guests`, `has_agent`, `lead_time_tier`, `country_group` (top 15
countries + "Other").

## Outputs (in `outputs/`)

PNG charts for every script, `gradient_boosting_feature_importance.csv`,
`gradient_boosting_predictions.csv`, `kmeans_final_clusters.csv`.

## The website

In the PyCharm **Terminal** (bottom of the window), inside the project folder:

    pip install -r requirements.txt
    streamlit run app.py

Your browser opens at http://localhost:8501 with 4 pages: predict a
booking, model comparison, insights, and the charts. The first start
trains the model (about 1 minute); after that it opens instantly.
Press Ctrl+C in the terminal to stop it.
