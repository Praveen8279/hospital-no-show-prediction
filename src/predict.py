import pandas as pd


def merge_external_schedule_data(base_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    external_df = external_df.copy()

    expected_cols = ["PatientId", "recent_cancel_count", "avg_delay_days", "department_load_score"]
    for col in expected_cols:
        if col not in external_df.columns:
            external_df[col] = 0

    merged = base_df.merge(
        external_df[expected_cols],
        on="PatientId",
        how="left"
    )

    merged["recent_cancel_count"] = merged["recent_cancel_count"].fillna(0)
    merged["avg_delay_days"] = merged["avg_delay_days"].fillna(0)
    merged["department_load_score"] = merged["department_load_score"].fillna(0)

    return merged