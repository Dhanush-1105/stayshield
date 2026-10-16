"""
StayShield website  (Streamlit)

Run from the PyCharm Terminal:      streamlit run app.py
It opens in your browser at         http://localhost:8501
"""
import json

import joblib
import pandas as pd
import sklearn
import streamlit as st

import config
from data_loader import load_data
from train_model import MODEL_PATH, RESULTS_PATH, train_and_save

st.set_page_config(page_title="StayShield", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Training the model for the first time (about 1 minute)...")
def get_model():
    # A saved model only works with the scikit-learn version that made it.
    # If it is missing, unreadable or from another version -> train again.
    try:
        bundle = joblib.load(MODEL_PATH)
        if bundle.get("sklearn_version") != sklearn.__version__:
            raise ValueError("scikit-learn version changed")
        return bundle
    except Exception:
        train_and_save()
        return joblib.load(MODEL_PATH)


@st.cache_data
def get_data():
    return load_data()


bundle = get_model()
data = get_data()

st.sidebar.title("🛡️ StayShield")
page = st.sidebar.radio("Go to", ["Predict a booking", "Model results",
                                  "Insights", "Clusters & charts"])
st.sidebar.caption(f"{len(data):,} hotel bookings analysed")

# ====================================================================== 1
if page == "Predict a booking":
    st.title("Will this booking be cancelled?")
    st.write("Enter the booking details. Anything not shown uses a typical value.")
    opt = bundle["options"]

    c1, c2, c3 = st.columns(3)
    hotel = c1.selectbox("Hotel", opt["hotel"])
    month = c1.selectbox("Arrival month", config.MONTH_ORDER, index=6)
    lead = c1.number_input("Lead time (days before arrival)", 0, 750, 60)
    wk_nights = c2.number_input("Week nights", 0, 30, 3)
    we_nights = c2.number_input("Weekend nights", 0, 20, 1)
    adults = c2.number_input("Adults", 1, 10, 2)
    children = c2.number_input("Children", 0, 10, 0)
    deposit = c3.selectbox("Deposit type", opt["deposit_type"])
    segment = c3.selectbox("Market segment", opt["market_segment"], index=
                           opt["market_segment"].index("Online TA"))
    customer = c3.selectbox("Customer type", opt["customer_type"])

    c4, c5, c6 = st.columns(3)
    adr = c4.number_input("Price per night (EUR)", 0.0, 600.0, 100.0)
    requests = c4.number_input("Special requests", 0, 5, 0)
    prev_cancel = c5.number_input("Previous cancellations", 0, 20, 0)
    changes = c5.number_input("Booking changes", 0, 20, 0)
    parking = c6.number_input("Car parking spaces", 0, 5, 0)
    country = c6.selectbox("Country", opt["country_group"])

    if st.button("Predict", type="primary"):
        row = dict(bundle["defaults"])
        nights = wk_nights + we_nights
        tier = "Short" if lead <= 30 else "Medium" if lead <= 120 else "Long"
        row.update({
            "hotel": hotel, "arrival_date_month": month, "lead_time": lead,
            "stays_in_week_nights": wk_nights, "stays_in_weekend_nights": we_nights,
            "total_nights": nights, "adults": adults, "children": children,
            "total_guests": adults + children, "deposit_type": deposit,
            "market_segment": segment, "customer_type": customer, "adr": adr,
            "total_of_special_requests": requests,
            "previous_cancellations": prev_cancel, "booking_changes": changes,
            "required_car_parking_spaces": parking, "country_group": country,
            "lead_time_tier": tier,
        })
        X_new = pd.DataFrame([row])[bundle["columns"]]
        prob = float(bundle["model"].predict_proba(X_new)[0, 1])

        level = "HIGH" if prob >= 0.6 else "MEDIUM" if prob >= 0.35 else "LOW"
        st.metric("Cancellation probability", f"{prob:.1%}")
        st.progress(prob)
        if level == "HIGH":
            st.error("High risk - consider asking for a deposit or reconfirming.")
        elif level == "MEDIUM":
            st.warning("Medium risk - worth monitoring.")
        else:
            st.success("Low risk - this booking is likely to stay.")

# ====================================================================== 2
elif page == "Model results":
    st.title("Model comparison")
    if not RESULTS_PATH.exists():
        train_and_save()
    res = pd.DataFrame(json.loads(RESULTS_PATH.read_text())).T
    st.dataframe(res.style.format("{:.3f}"), width="stretch")
    st.bar_chart(res["accuracy"])
    st.caption("Precision/recall/F1 are for the 'cancelled' class. "
               "Models were tested on 20% of bookings they never saw.")

# ====================================================================== 3
elif page == "Insights":
    st.title("What drives cancellations?")
    st.metric("Overall cancellation rate", f"{data[config.TARGET].mean():.1%}")

    col = st.selectbox("Cancellation rate by...", [
        "deposit_type", "market_segment", "hotel", "customer_type",
        "lead_time_tier", "arrival_date_month", "country_group"])
    rate = data.groupby(col)[config.TARGET].mean().sort_values(ascending=False)
    if col == "arrival_date_month":
        rate = rate.reindex(config.MONTH_ORDER)
    st.bar_chart(rate)

    imp_path = config.OUTPUT_DIR / "gradient_boosting_feature_importance.csv"
    if imp_path.exists():
        st.subheader("Top features (Gradient Boosting)")
        imp = pd.read_csv(imp_path).head(15).set_index("Feature")
        st.bar_chart(imp)

# ====================================================================== 4
else:
    st.title("Clusters & charts")
    st.write("Charts created by the scripts (run `run_all.py` to refresh them).")
    pngs = sorted(config.OUTPUT_DIR.glob("*.png"))
    if not pngs:
        st.info("No charts yet - run run_all.py first.")
    for i in range(0, len(pngs), 2):
        cols = st.columns(2)
        for c, p in zip(cols, pngs[i:i + 2]):
            c.image(str(p), caption=p.stem.replace("_", " "))
