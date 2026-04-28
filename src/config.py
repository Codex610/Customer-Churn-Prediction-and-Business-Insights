import os

# ─── Paths ───────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW        = os.path.join(BASE_DIR, "data", "raw", "telco_churn.csv")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed", "churn_clean.csv")

MODEL_PATH      = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH     = os.path.join(BASE_DIR, "models", "scaler.pkl")

REPORTS_DIR     = os.path.join(BASE_DIR, "reports")

# ─── Target ──────────────────────────────────────────────
TARGET = "Churn"

# ─── Numerical columns ───────────────────────────────────
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# ─── Categorical columns ─────────────────────────────────
CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

# ─── Train / Test split ──────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ─── Models to train ─────────────────────────────────────
MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

# ─── Revenue model ───────────────────────────────────────
AVG_MONTHLY_REVENUE   = 65.0
AVG_CUSTOMER_LIFETIME = 24

# ─── MLflow ──────────────────────────────────────────────
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/codex03080/Customer-Churn-Prediction-and-Business-Insights.mlflow")
MLFLOW_EXPERIMENT_NAME = "telco_churn"
