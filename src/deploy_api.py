import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from .app_config import MODEL_PATH, SCALER_PATH

# ─── Load model & scaler once at startup ─────────────────
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="Telco Churn Prediction API")


# ─── Request schema ──────────────────────────────────────
class CustomerInput(BaseModel):
    tenure:           float
    MonthlyCharges:   float
    TotalCharges:     float
    # Add more fields as needed — must match training features
    SeniorCitizen:    int   = 0
    Partner:          int   = 0
    Dependents:       int   = 0
    PhoneService:     int   = 1
    PaperlessBilling: int   = 0


# ─── Routes ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API is running."}


@app.post("/predict")
def predict(customer: CustomerInput):
    """Predict churn probability for a single customer."""
    input_df = pd.DataFrame([customer.dict()])

    # Scale numerical columns
    from .app_config import NUM_COLS
    num_present = [c for c in NUM_COLS if c in input_df.columns]
    input_df[num_present] = scaler.transform(input_df[num_present])

    # Align columns with training data
    model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else input_df.columns
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]

    churn_prob = round(float(model.predict_proba(input_df)[0][1]), 4)
    churn_pred = int(model.predict(input_df)[0])

    return {
        "churn_prediction": churn_pred,
        "churn_probability": churn_prob,
        "risk_level": "High" if churn_prob > 0.6 else "Medium" if churn_prob > 0.3 else "Low"
    }


# ─── Run ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deploy_api:app", host="0.0.0.0", port=8000, reload=True)
