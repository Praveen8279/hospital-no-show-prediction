from sklearn.model_selection import train_test_split

from src.data_ingestion import load_data
from src.preprocessing import clean_data
from src.feature_engineering import add_history_features, get_model_data
from src.eda import run_eda
from src.train import train_models
from src.explain import generate_shap_outputs


def main():
    df = load_data()
    df = clean_data(df)

    for col in ["recent_cancel_count", "avg_delay_days", "department_load_score"]:
        if col not in df.columns:
            df[col] = 0

    df = add_history_features(df)

    run_eda(df)

    X, y, feature_cols = get_model_data(df)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    best_model, metadata = train_models(
        X_train, X_valid, X_test, y_train, y_valid, y_test
    )

    sample_size = min(1000, len(X_test))
    generate_shap_outputs(X_test.sample(sample_size, random_state=42))


if __name__ == "__main__":
    main()