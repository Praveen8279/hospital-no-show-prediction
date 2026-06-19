import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def build_preprocessor(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }
    return model, metrics


def tune_xgboost(X_train, y_train, X_valid, y_valid, preprocessor):
    def objective(trial):
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", XGBClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 100, 400),
                    max_depth=trial.suggest_int("max_depth", 3, 8),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    subsample=trial.suggest_float("subsample", 0.7, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1,
                )),
            ]
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params


def tune_lightgbm(X_train, y_train, X_valid, y_valid, preprocessor):
    def objective(trial):
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LGBMClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 100, 400),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 15, 100),
                    subsample=trial.suggest_float("subsample", 0.7, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )),
            ]
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params


def train_models(X_train, X_valid, X_test, y_train, y_valid, y_test):
    preprocessor = build_preprocessor(X_train)

    best_xgb_params = tune_xgboost(X_train, y_train, X_valid, y_valid, preprocessor)
    best_lgbm_params = tune_lightgbm(X_train, y_train, X_valid, y_valid, preprocessor)

    xgb_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                **best_xgb_params,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            )),
        ]
    )

    lgbm_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(
                **best_lgbm_params,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )),
        ]
    )

    trained_xgb, xgb_metrics = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")
    trained_lgbm, lgbm_metrics = evaluate_model(lgbm_model, X_train, X_test, y_train, y_test, "LightGBM")

    metrics = [xgb_metrics, lgbm_metrics]
    best_metrics = max(metrics, key=lambda x: x["roc_auc"])
    best_model = trained_xgb if best_metrics["model"] == "XGBoost" else trained_lgbm

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    metadata = {
        "best_model": best_metrics["model"],
        "metrics": metrics,
        "best_params": {
            "xgboost": best_xgb_params,
            "lightgbm": best_lgbm_params,
        },
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
    }

    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return best_model, metadata