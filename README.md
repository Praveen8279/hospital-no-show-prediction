Hospital Appointment No-Show Prediction
An end-to-end healthcare machine learning project that predicts whether a patient is likely to miss a scheduled hospital appointment using patient history, scheduling behavior, and optional external scheduling signals.

Overview
This project builds a full machine learning workflow for hospital appointment no-show prediction, including preprocessing, feature engineering, model training, explainability, and a Streamlit web app for interactive risk scoring.

The app can be deployed directly from GitHub to Streamlit Community Cloud, which supports selecting a repository, branch, and main app file during deployment.

Features
Data cleaning and preprocessing for medical appointment records.

Feature engineering for wait time, prior attendance behavior, chronic conditions, weekday patterns, and optional external scheduling signals.

Model training with advanced boosting models such as XGBoost and LightGBM, both available through Python APIs compatible with scikit-learn style workflows.

Hyperparameter tuning with Optuna.

SHAP-based explainability for tree-based models.

Streamlit app for interactive prediction and CSV upload support.

Docker support for containerized local or cloud deployment.

Architecture
GitHub supports Mermaid diagrams inside Markdown files, so the following architecture diagram can be shown directly in the repository README.




Project Structure
text
hospital-no-show-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
├── notebooks/
│   └── 01_eda.ipynb
├── outputs/
│   ├── metrics.json
│   ├── shap_summary_bar.png
│   └── shap_summary_beeswarm.png
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── explain.py
├── .gitignore
├── .dockerignore
├── Dockerfile
├── app.py
├── main.py
├── requirements.txt
└── README.md
Tech Stack
Python

pandas

NumPy

scikit-learn

XGBoost.

LightGBM.

SHAP.

Optuna.

Streamlit.

Docker.

How to Run Locally
1. Clone the repository
bash
git clone https://github.com/Praveen8279/hospital-no-show-prediction.git
cd hospital-no-show-prediction
2. Create and activate a virtual environment
bash
python -m venv .venv
Windows:

bash
.venv\Scripts\activate
macOS/Linux:

bash
source .venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Run training
bash
python main.py
5. Launch the Streamlit app
bash
streamlit run app.py