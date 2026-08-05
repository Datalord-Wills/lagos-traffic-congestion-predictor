import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime, date, time as dtime

st.set_page_config(
    page_title="Lagos Traffic Congestion Predictor",
    page_icon="🚦",
    layout="centered",
)

# ----------------------------------------------------------------------
# LOAD MODEL + SEGMENT METADATA
# ----------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_segment_meta():
    return pd.read_csv("segment_metadata.csv")

model = load_model()
segment_meta = load_segment_meta()

LABEL_MAP = {1: "Low", 2: "Normal", 3: "High", 4: "Heavy"}
LABEL_COLOR = {
    "Low": "#2ecc71",
    "Normal": "#f1c40f",
    "High": "#e67e22",
    "Heavy": "#e74c3c",
}
LABEL_ADVICE = {
    "Low": "Roads are clear. Good time to travel.",
    "Normal": "Typical traffic. No major delays expected.",
    "High": "Significant slowdown likely. Consider an alternate route or time.",
    "Heavy": "Severe congestion expected. Delay your trip or use an alternate route if possible.",
}

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------

st.title("🚦 Lagos Traffic Congestion Predictor")
st.caption(
    "A web interface for the trained Random Forest congestion model "
    "(F1-macro ≈ 0.947 on held-out test data) developed for the 3MTT "
    "NextGen capstone project."
)

st.divider()

# ----------------------------------------------------------------------
# INPUT FORM
# ----------------------------------------------------------------------

st.subheader("1. Road & time")

col1, col2 = st.columns(2)
with col1:
    segment = st.selectbox("Road segment", segment_meta["Segment"].tolist())
with col2:
    trip_date = st.date_input("Date", value=date.today())

trip_time = st.time_input("Time", value=dtime(8, 0))

st.subheader("2. Estimated vehicle counts (this 15-min window)")
st.caption(
    "The model was trained on 15-minute vehicle counts per segment. "
    "If you don't have a live count, use the 'Typical for this hour' "
    "button below to auto-fill a reasonable estimate based on the "
    "training data's average pattern for the selected segment and hour."
)

seg_row = segment_meta[segment_meta["Segment"] == segment].iloc[0]

if "car_count" not in st.session_state:
    st.session_state.car_count = 80
    st.session_state.keke_count = 60
    st.session_state.bus_count = 30
    st.session_state.truck_count = 15

col3, col4 = st.columns(2)
with col3:
    car_count = st.number_input("Car count", min_value=0, max_value=500,
                                 value=st.session_state.car_count, step=1)
    bus_count = st.number_input("Bus count", min_value=0, max_value=500,
                                 value=st.session_state.bus_count, step=1)
with col4:
    keke_count = st.number_input("Keke/Okada count", min_value=0, max_value=500,
                                  value=st.session_state.keke_count, step=1)
    truck_count = st.number_input("Truck count", min_value=0, max_value=500,
                                   value=st.session_state.truck_count, step=1)

st.subheader("3. Conditions")

col5, col6 = st.columns(2)
with col5:
    is_market_day = st.checkbox("Market day (e.g. Tuesday / Friday market axis)")
    is_school_hours = st.checkbox("School run hours (7-8am or 2-3pm on a school day)")
with col6:
    is_fuel_scarcity = st.checkbox("Fuel scarcity currently ongoing")

is_raining = st.checkbox("Raining")
rainfall_intensity = 0.0
if is_raining:
    rainfall_intensity = st.slider(
        "Rainfall intensity", min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="0.1 = light drizzle, 1.0 = heavy downpour"
    )

st.divider()

# ----------------------------------------------------------------------
# PREDICT
# ----------------------------------------------------------------------

if st.button("Predict congestion", type="primary", use_container_width=True):

    dt = datetime.combine(trip_date, trip_time)
    hour = dt.hour
    minute = dt.minute
    day_of_week = dt.strftime("%A")
    is_weekend = int(day_of_week in ["Saturday", "Sunday"])
    day_of_month = dt.day
    month = dt.month
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Column order must exactly match the order the model was trained on
    input_row = pd.DataFrame([{
        "Day_of_week": day_of_week,
        "Segment": segment,
        "Road_Type": seg_row["Road_Type"],
        "Lanes": seg_row["Lanes"],
        "CarCount": car_count,
        "KekeOkadaCount": keke_count,
        "BusCount": bus_count,
        "TruckCount": truck_count,
        "Is_Market_Day": int(is_market_day),
        "Is_School_Hours": int(is_school_hours),
        "Rainfall_Intensity": rainfall_intensity,
        "Is_Fuel_Scarcity": int(is_fuel_scarcity),
        "Day_of_month": day_of_month,
        "Month": month,
        "Is_weekend": is_weekend,
        "Minute": minute,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }])

    pred_code = model.predict(input_row)[0]
    pred_label = LABEL_MAP[pred_code]
    pred_proba = model.predict_proba(input_row)[0]
    classes = [LABEL_MAP[c] for c in model.named_steps["randomforestclassifier"].classes_]

    color = LABEL_COLOR[pred_label]
    st.markdown(
        f"""
        <div style="background-color:{color}22; border-left: 6px solid {color};
                    padding: 1.2rem; border-radius: 8px; margin-top: 1rem;">
            <h3 style="margin:0; color:{color};">Predicted congestion: {pred_label}</h3>
            <p style="margin:0.4rem 0 0 0;">{LABEL_ADVICE[pred_label]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.caption("Prediction confidence by class")

    proba_df = pd.DataFrame({
        "Class": classes,
        "Probability": pred_proba,
    })
    proba_df["Percent"] = (proba_df["Probability"] * 100).round(1)
    proba_df["Label"] = proba_df["Percent"].astype(str) + "%"

    # Keep class order fixed as Low, Normal, High, Heavy regardless of
    # which one is highest, so the chart reads consistently every time
    class_order = ["Low", "Normal", "High", "Heavy"]
    bar_colors = [LABEL_COLOR[c] for c in class_order]

    bars = (
        alt.Chart(proba_df)
        .mark_bar()
        .encode(
            x=alt.X("Class:N", sort=class_order, title=None),
            y=alt.Y("Percent:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Class:N",
                sort=class_order,
                scale=alt.Scale(domain=class_order, range=bar_colors),
                legend=None,
            ),
        )
    )

    labels = (
        alt.Chart(proba_df)
        .mark_text(dy=-8, fontSize=14, fontWeight="bold")
        .encode(
            x=alt.X("Class:N", sort=class_order),
            y=alt.Y("Percent:Q"),
            text="Label:N",
        )
    )

    st.altair_chart((bars + labels).properties(height=300), use_container_width=True)

st.divider()
st.caption(
    "⚠️ Trained on a synthetically generated Lagos traffic dataset "
    "(46,080 rows, 8 segments, 60 days), not live sensor data. "
    "Vehicle counts are required inputs because the model was trained "
    "with them as its strongest predictors — in a future version with "
    "live sensor or API-fed volume data, these could be filled in "
    "automatically instead of estimated by the user."
)