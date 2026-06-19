import pandas as pd
import numpy as np

def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["PatientId", "ScheduledDay", "AppointmentDay"]).reset_index(drop=True)

    df["prior_appointments"] = df.groupby("PatientId").cumcount()

    df["prior_no_shows"] = (
        df.groupby("PatientId")["no_show"]
        .transform(lambda s: s.shift(1).fillna(0).cumsum())
    )

    df["prior_shows"] = (
        df.groupby("PatientId")["no_show"]
        .transform(lambda s: (1 - s.shift(1).fillna(0)).cumsum())
    )

    df["prior_show_rate"] = np.where(
        df["prior_appointments"] > 0,
        df["prior_shows"] / df["prior_appointments"],
        0
    )

    df["chronic_count"] = (
        df["Hipertension"].fillna(0)
        + df["Diabetes"].fillna(0)
        + df["Alcoholism"].fillna(0)
        + df["Handcap"].fillna(0)
    )

    df["is_weekend_booking"] = df["scheduled_weekday"].isin(["Saturday", "Sunday"]).astype(int)
    df["appointment_month"] = df["AppointmentDay"].dt.month
    df["scheduled_hour"] = df["ScheduledDay"].dt.hour
    df["is_same_day"] = (df["days_wait"] == 0).astype(int)

    df["age_group"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 35, 50, 65, 120],
        labels=["child", "teen", "young_adult", "adult", "middle_age", "senior"]
    )

    return df


def get_model_data(df: pd.DataFrame):
    feature_cols = [
        "Gender",
        "Age",
        "age_group",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received",
        "days_wait",
        "appointment_weekday",
        "scheduled_weekday",
        "Neighbourhood",
        "prior_appointments",
        "prior_no_shows",
        "prior_show_rate",
        "chronic_count",
        "is_weekend_booking",
        "appointment_month",
        "scheduled_hour",
        "is_same_day",
    ]

    X = df[feature_cols]
    y = df["no_show"]
    return X, y, feature_cols