from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


def generate_shap_outputs(X_sample: pd.DataFrame):
    pipeline = joblib.load(MODELS_DIR / "best_model.joblib")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_sample)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        plot_type="bar"
    )
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_summary_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()