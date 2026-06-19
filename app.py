import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/best_model.joblib")
METADATA_PATH = Path("models/model_metadata.json")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata


def get_age_group(age):
    if age <= 12:
        return "child"
    elif age <= 18:
        return "teen"
    elif age <= 35:
        return "young_adult"
    elif age <= 50:
        return "adult"
    elif age <= 65:
        return "middle_age"
    return "senior"


def get_risk_band(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    return "High Risk"


def build_input_dataframe(inputs):
    return pd.DataFrame([inputs])


st.set_page_config(
    page_title="Hospital No-Show Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Appointment No-Show Prediction")
st.caption("Healthcare ML project for predicting whether a patient may miss a scheduled appointment.")

model, metadata = load_artifacts()

col_a, col_b, col_c = st.columns(3)
col_a.metric("Best Model", metadata.get("best_model", "N/A"))
col_b.metric("Train Rows", metadata.get("train_rows", "N/A"))
col_c.metric("Test Rows", metadata.get("test_rows", "N/A"))

st.markdown("---")

uploaded_file = st.file_uploader("Upload external scheduling CSV (optional)", type=["csv"])

external_features = {
    "recent_cancel_count": 0,
    "avg_delay_days": 0,
    "department_load_score": 0,
}

if uploaded_file is not None:
    ext_df = pd.read_csv(uploaded_file)
    st.subheader("External scheduling data preview")
    st.dataframe(ext_df.head(), use_container_width=True)

    required_cols = {"PatientId", "recent_cancel_count", "avg_delay_days", "department_load_score"}
    if required_cols.issubset(ext_df.columns):
        patient_id_lookup = st.number_input("PatientId for external lookup", min_value=0, value=1)
        matched = ext_df[ext_df["PatientId"] == patient_id_lookup]
        if not matched.empty:
            row = matched.iloc[0]
            external_features = {
                "recent_cancel_count": float(row["recent_cancel_count"]),
                "avg_delay_days": float(row["avg_delay_days"]),
                "department_load_score": float(row["department_load_score"]),
            }

with st.form("prediction_form"):
    st.subheader("Patient and Appointment Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        patient_id = st.number_input("PatientId", min_value=0, value=1)
        gender = st.selectbox("Gender", ["F", "M"])
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        scholarship = st.selectbox("Scholarship", [0, 1])
        hipertension = st.selectbox("Hipertension", [0, 1])
        diabetes = st.selectbox("Diabetes", [0, 1])
        alcoholism = st.selectbox("Alcoholism", [0, 1])
        handcap = st.number_input("Handcap", min_value=0, max_value=4, value=0)

    with col2:
        sms_received = st.selectbox("SMS Received", [0, 1])
        days_wait = st.number_input("Days Wait", min_value=0, max_value=365, value=7)
        appointment_weekday = st.selectbox(
            "Appointment Weekday",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        )
        scheduled_weekday = st.selectbox(
            "Scheduled Weekday",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        neighbourhood = st.text_input("Neighbourhood", value="JARDIM CAMBURI")
        appointment_month = st.slider("Appointment Month", 1, 12, 5)
        scheduled_hour = st.slider("Scheduled Hour", 0, 23, 10)

    with col3:
        prior_appointments = st.number_input("Prior Appointments", min_value=0, max_value=100, value=0)
        prior_no_shows = st.number_input("Prior No-Shows", min_value=0, max_value=100, value=0)
        chronic_count = st.number_input("Chronic Count", min_value=0, max_value=10, value=0)
        is_weekend_booking = st.selectbox("Weekend Booking", [0, 1])
        is_same_day = st.selectbox("Same Day Appointment", [0, 1])

    submitted = st.form_submit_button("Predict No-Show Risk")

if submitted:
    age_group = get_age_group(age)
    prior_show_rate = 0 if prior_appointments == 0 else (prior_appointments - prior_no_shows) / prior_appointments

    input_data = {
        "PatientId": patient_id,
        "Gender": gender,
        "Age": age,
        "age_group": age_group,
        "Scholarship": scholarship,
        "Hipertension": hipertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handcap": handcap,
        "SMS_received": sms_received,
        "days_wait": days_wait,
        "appointment_weekday": appointment_weekday,
        "scheduled_weekday": scheduled_weekday,
        "Neighbourhood": neighbourhood,
        "prior_appointments": prior_appointments,
        "prior_no_shows": prior_no_shows,
        "prior_show_rate": prior_show_rate,
        "chronic_count": chronic_count,
        "is_weekend_booking": is_weekend_booking,
        "appointment_month": appointment_month,
        "scheduled_hour": scheduled_hour,
        "is_same_day": is_same_day,
        "recent_cancel_count": external_features["recent_cancel_count"],
        "avg_delay_days": external_features["avg_delay_days"],
        "department_load_score": external_features["department_load_score"],
    }

    input_df = build_input_dataframe(input_data)

    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(model.predict(input_df)[0])
    risk_band = get_risk_band(probability)

    st.markdown("---")
    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("No-Show Probability", f"{probability:.2%}")
    result_col2.metric("Risk Band", risk_band)
    result_col3.metric("Predicted Label", "No-Show" if prediction == 1 else "Will Attend")

    st.progress(min(max(probability, 0.0), 1.0))

    if risk_band == "High Risk":
        st.error("High no-show risk detected. This appointment may need proactive follow-up.")
    elif risk_band == "Medium Risk":
        st.warning("Moderate no-show risk detected. Consider reminder reinforcement.")
    else:
        st.success("Low no-show risk detected. Attendance is more likely.")

    st.subheader("Submitted Features")
    st.dataframe(input_df, use_container_width=True)