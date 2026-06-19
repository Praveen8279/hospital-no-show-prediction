import pandas as pd

def clean_appointments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [col.strip() for col in df.columns]

    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

    df["no_show"] = df["No-show"].map({"No": 0, "Yes": 1})

    df["days_wait"] = (
        df["AppointmentDay"].dt.normalize()
        - df["ScheduledDay"].dt.normalize()
    ).dt.days

    df["appointment_weekday"] = df["AppointmentDay"].dt.day_name()
    df["scheduled_weekday"] = df["ScheduledDay"].dt.day_name()

    df = df[df["Age"] >= 0]
    df = df[df["days_wait"] >= 0]

    return df