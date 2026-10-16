"""
StayShield - central settings.

Every script imports this file, so all paths, column names and options
live in ONE place. Nothing in the project uses absolute (C:\\...) paths,
so it runs from any folder on any computer.
"""
import os
from pathlib import Path

import matplotlib

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_CSV = DATA_DIR / "hotel_bookings.csv"
PREPROCESSED_CSV = DATA_DIR / "hotel_bookings_preprocessed.csv"

# ---------------------------------------------------------------- run mode
# Run a script directly in PyCharm  -> plots pop up, questions are asked.
# run_all.py sets these to "0"      -> everything runs without stopping.
SHOW_PLOTS = os.environ.get("STAYSHIELD_SHOW_PLOTS", "1") == "1"
INTERACTIVE = os.environ.get("STAYSHIELD_INTERACTIVE", "1") == "1"

if not SHOW_PLOTS:
    matplotlib.use("Agg")  # draw to files only, no windows

# ---------------------------------------------------------------- general
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------- targets
TARGET = "is_canceled"   # classification target (1 = booking was cancelled)
REG_TARGET = "adr"       # regression target (average daily rate, in euros)

# ---------------------------------------------------------------- columns
# These two columns are only known AFTER the guest cancels / checks out.
# reservation_status literally says "Canceled", so keeping it = cheating.
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]

# Replaced by engineered columns (has_agent, country_group), so removed.
REPLACED_COLUMNS = ["agent", "company", "country"]

# Columns used to demonstrate scaling (min_max.py, standard.py, outlier.py)
SCALE_COLS = [
    "lead_time",
    "adr",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_of_special_requests",
]

# Nominal (no order) columns -> One-Hot encoding
NOMINAL_COLS = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "customer_type",
    "reserved_room_type",
    "assigned_room_type",
    "country_group",
]

# Ordinal (has an order) columns -> Ordinal encoding. Order is DECLARED,
# never guessed by the encoder.
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
LEAD_TIME_ORDER = ["Short", "Medium", "Long"]
ORDINAL_COLS = ["arrival_date_month", "lead_time_tier"]
ORDINAL_ORDERS = [MONTH_ORDER, LEAD_TIME_ORDER]

# Features for Logistic Regression (classification)
LOGISTIC_FEATURES = [
    "lead_time",
    "previous_cancellations",
    "booking_changes",
    "total_of_special_requests",
    "required_car_parking_spaces",
    "days_in_waiting_list",
    "total_nights",
    "adr",
]

# Features for Linear Regression (predicting adr)
LINEAR_FEATURES = [
    "lead_time",
    "total_nights",
    "adults",
    "children",
    "total_of_special_requests",
    "arrival_date_week_number",
]

# Clustering
CLUSTER_2D_FEATURES = ["lead_time", "adr"]       # k-means 2-D demo
PLOT_X, PLOT_Y = "lead_time", "adr"              # axes for the DBSCAN plot
CLUSTER_DROP_COLUMNS = [TARGET, "IsAnomaly"]     # never cluster on these
ANOMALY_COLS = ["adr", "lead_time"]              # IQR rule -> IsAnomaly flag
DBSCAN_SAMPLE_SIZE = 1000
DBSCAN_MIN_SAMPLES = 5
