# 📉 Telco Customer Churn Prediction

> **End-to-end Machine Learning project** — Predict which customers are about to leave, estimate the revenue at risk, and explain *why* they're churning.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Business Problem](#-business-problem)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Quick Start](#-quick-start)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Feature Engineering](#-feature-engineering)
- [Models Trained](#-models-trained)
- [Evaluation Results](#-evaluation-results)
- [Revenue Impact Logic](#-revenue-impact-logic)
- [SHAP Explainability](#-shap-explainability)
- [API Usage](#-api-usage)
- [Reports & Outputs](#-reports--outputs)
- [Production Monitoring](#-production-monitoring)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)

---

## 🔭 Overview

This project builds a **customer churn prediction system** for a telecommunications company using the Telco Customer Churn dataset. The pipeline goes from raw CSV to a deployed REST API in a single command.

**What it does:**
- Predicts whether a customer will churn (leave) in the next month
- Estimates expected monthly revenue loss per customer
- Identifies **priority intervention targets** (high-risk + high-value customers)
- Explains model decisions using SHAP values
- Serves predictions through a FastAPI REST endpoint

---

## 💼 Business Problem

Customer churn is one of the most expensive problems in telecom. Acquiring a new customer costs **5–7× more** than retaining an existing one. A model that accurately identifies at-risk customers — before they leave — allows the retention team to intervene with targeted offers.

**Goal:** Given a customer's profile and usage data, predict the probability they will churn, and quantify the revenue at stake.

---

## 🗂️ Project Structure

```
customer_churn_project/
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv          ← place your dataset here
│   │
│   └── processed/
│       └── churn_clean.csv          ← auto-generated after pipeline run
│
├── notebooks/
│   └── 01_eda.ipynb                 ← exploratory data analysis
│
├── src/
│   │
│   ├── config.py                    # Global config (paths, parameters)
│   ├── data_loader.py               # Load & clean dataset
│   ├── preprocessing.py             # Missing values + encoding + scaling
│   ├── feature_engineering.py       # Tenure buckets, interaction features, etc.
│   ├── train.py                     # Train base models
│   ├── tune.py                      # GridSearch + Optuna tuning
│   ├── evaluate.py                  # Metrics + ROC + confusion matrix
│   ├── revenue_model.py             # Revenue loss estimation logic
│   ├── shap_analysis.py             # SHAP interpretation
│   ├── deploy_api.py                # FastAPI app
│   └── main.py                      # End-to-end pipeline runner
│
├── models/
│   ├── best_model.pkl               ← saved after pipeline run
│   └── scaler.pkl                   ← saved after pipeline run
│
├── reports/
│   ├── metrics.json                 ← all model metrics
│   ├── model_comparison.csv         ← sortable comparison table
│   ├── shap_summary.png             ← SHAP beeswarm plot
│   └── revenue_impact.csv           ← per-customer revenue risk
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📦 Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target** | `Churn` — Yes / No |
| **Churn rate** | ~26.5% (class imbalance handled via class weights) |

### Key Columns

| Column | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months the customer has been with the company |
| `MonthlyCharges` | Numeric | Current monthly bill |
| `TotalCharges` | Numeric | Total amount charged to date |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / etc. |
| `Churn` | Target | Whether the customer left (Yes / No) |

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer_churn_project.git
cd customer_churn_project
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), rename it to `telco_churn.csv`, and place it at:

```
data/raw/telco_churn.csv
```

### 5. Run the full pipeline

```bash
python src/main.py
```

This single command runs all 9 steps and produces every output in `models/` and `reports/`.

### 6. (Optional) Launch the prediction API

```bash
uvicorn src.deploy_api:app --reload --port 8000
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 🔄 Pipeline Walkthrough

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py                                  │
│                                                                 │
│  [1] data_loader        → Load raw CSV, basic cleaning          │
│       ↓                                                         │
│  [2] feature_engineering → Add 4 new domain features           │
│       ↓                                                         │
│  [3] preprocessing      → Encode + Scale + Train/Test Split     │
│       ↓                                                         │
│  [4] train              → Fit LR, Random Forest, XGBoost        │
│       ↓                                                         │
│  [5] tune               → GridSearchCV + Optuna optimization    │
│       ↓                                                         │
│  [6] evaluate           → Metrics, ROC curves, comparison table │
│       ↓                                                         │
│  [7] revenue_model      → Expected revenue loss per customer    │
│       ↓                                                         │
│  [8] shap_analysis      → Feature importance & SHAP plots       │
│       ↓                                                         │
│  [9] Save best_model.pkl + scaler.pkl                           │
└─────────────────────────────────────────────────────────────────┘
```

Each step is a clean, independent module — easy to debug or re-run individually.

---

## 🛠️ Feature Engineering

Four domain-driven features are added before model training:

| Feature | Logic | Why It Matters |
|---|---|---|
| `tenure_bucket` | 0–12 months = Short (0), 13–36 = Mid (1), 37+ = Long (2) | Churn risk is highest in the first year |
| `total_services` | Count of active add-ons (phone, internet, streaming, etc.) | More services = higher switching cost = lower churn |
| `is_high_value` | `MonthlyCharges > $70` → 1, else 0 | Prioritise high-revenue customers for retention |
| `contract_tenure` | Contract type (numeric) × tenure | Long-term contract + long tenure = very low churn risk |

---

## 🤖 Models Trained

| Model | Type | Imbalance Handling |
|---|---|---|
| **Logistic Regression** | Linear baseline | `class_weight='balanced'` |
| **Random Forest** | Ensemble (bagging) | `class_weight='balanced'` |
| **XGBoost** | Gradient boosting | `scale_pos_weight = neg/pos ratio` |
| **LR (Tuned)** | GridSearchCV over C, penalty | Same as above |
| **XGBoost (Tuned)** | Optuna Bayesian search (30 trials) | Same as above |

### Why class weights?

The dataset has ~26% churners and ~74% non-churners. Without correction, a model can achieve 74% accuracy by just predicting "No churn" for everyone. Class weights force the model to pay more attention to the minority class (churners) during training.

---

## 📊 Evaluation Results

### Example `metrics.json`

```json
[
  {
    "model": "logistic_regression",
    "accuracy": 0.8012,
    "precision": 0.6421,
    "recall": 0.5518,
    "f1": 0.5935,
    "roc_auc": 0.8498,
    "confusion_matrix": [[916, 97], [182, 222]]
  },
  {
    "model": "random_forest",
    "accuracy": 0.7981,
    "precision": 0.6589,
    "recall": 0.4972,
    "f1": 0.5667,
    "roc_auc": 0.8401,
    "confusion_matrix": [[934, 79], [205, 199]]
  },
  {
    "model": "xgboost",
    "accuracy": 0.8143,
    "precision": 0.6712,
    "recall": 0.5615,
    "f1": 0.6115,
    "roc_auc": 0.8631,
    "confusion_matrix": [[921, 92], [179, 225]]
  },
  {
    "model": "xgboost_tuned",
    "accuracy": 0.8198,
    "precision": 0.6804,
    "recall": 0.5739,
    "f1": 0.6226,
    "roc_auc": 0.8714,
    "confusion_matrix": [[924, 89], [173, 231]]
  }
]
```

### Reading the Confusion Matrix

```
                   Predicted: No    Predicted: Yes
Actual: No  (TN)      924               89       ← False Positives (unnecessary offers)
Actual: Yes (FN)      173              231       ← False Negatives (missed churners)
```

> **For churn prediction, Recall matters most.** A false negative (missing a churner) is far more expensive than a false positive (offering a discount to someone who wasn't leaving).

### Model Selection Criterion

The best model is selected by **ROC-AUC** — it measures ranking ability across all thresholds and is robust to class imbalance.

---

## 💰 Revenue Impact Logic

### Formula

```
Expected Monthly Revenue Loss = P(churn) × MonthlyCharges
```

### Why this formula?

When a customer churns, they stop paying their monthly bill. Multiplying the **probability** of churn by the **monthly charge** gives the **expected value** of revenue at risk — a dollar figure the business team can act on without waiting for actual churn to happen.

### Priority Customer Definition

A customer is flagged as a **priority intervention target** when they are both:
- **High risk:** churn probability ≥ 0.6
- **High value:** MonthlyCharges ≥ $70

These customers are sorted to the top of `revenue_impact.csv` for the retention team.

### Example Output (`revenue_impact.csv`)

| customer_idx | churn_probability | monthly_charges | expected_revenue_loss | high_risk | high_value | priority_customer |
|---|---|---|---|---|---|---|
| 1034 | 0.8412 | 99.65 | 83.81 | 1 | 1 | 1 |
| 2871 | 0.7901 | 84.80 | 66.99 | 1 | 1 | 1 |
| 445  | 0.7234 | 70.70 | 51.14 | 1 | 1 | 1 |
| 3102 | 0.6891 | 55.20 | 38.04 | 1 | 0 | 0 |
| 789  | 0.3211 | 95.10 | 30.55 | 0 | 1 | 0 |

### Aggregate Summary (printed to console)

```
[revenue_model] ── Revenue Impact Summary ──
  Total expected monthly revenue at risk : $12,847.33
  Priority customers (high risk + value) : 87
  Revenue at risk from priority customers: $7,214.55
```

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) tells us **why** the model made each prediction — not just which features are globally important.

### How to read the SHAP summary plot (`shap_summary.png`)

```
Feature            Impact on churn probability
─────────────────────────────────────────────
tenure             ◀── long tenure → lower churn (blue dots left)
Contract_Two year  ◀── long contract → much lower churn
MonthlyCharges     ──▶ high charges → higher churn (red dots right)
InternetService    ──▶ Fiber optic → higher churn risk
total_services     ◀── more services → lower churn
OnlineSecurity     ◀── has security → lower churn
tenure_bucket      ◀── longer tenure bucket → lower churn
```

- **X-axis:** SHAP value — positive means "pushed toward churn", negative means "pushed away from churn"
- **Color:** Feature value — red = high, blue = low
- **Each dot** = one customer in the test set

### Top Churn Drivers (typical output)

```
          feature  mean_abs_shap
           tenure       0.412310
 Contract_Two year       0.318240
   MonthlyCharges       0.287650
 InternetService_Fiber   0.201440
   total_services       0.154320
    tenure_bucket       0.134210
  OnlineSecurity        0.118760
```

---

## 🔌 API Usage

### Start the API

```bash
uvicorn src.deploy_api:app --reload --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Predict churn for one customer |
| `GET` | `/docs` | Interactive Swagger UI |

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "XGBClassifier"
}
```

---

### `POST /predict`

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 75.50,
    "TotalCharges": 906.00,
    "SeniorCitizen": 0,
    "gender": 0,
    "Partner": 1,
    "Dependents": 0,
    "PhoneService": 1,
    "MultipleLines": 0,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 1,
    "PaperlessBilling": 1,
    "tenure_bucket": 0,
    "total_services": 4,
    "is_high_value": 1,
    "contract_tenure": 0
  }'
```

**Response:**
```json
{
  "churn_probability": 0.7821,
  "churn_prediction": 1,
  "risk_level": "High"
}
```

### Risk Level Mapping

| Probability | Risk Level | Recommended Action |
|---|---|---|
| < 0.30 | 🟢 Low | No action needed |
| 0.30 – 0.59 | 🟡 Medium | Monitor closely |
| ≥ 0.60 | 🔴 High | Immediate retention outreach |

---

## 📁 Reports & Outputs

After running `python src/main.py`, the following files are generated:

| File | Description |
|---|---|
| `models/best_model.pkl` | Serialised best-performing model |
| `models/scaler.pkl` | Fitted StandardScaler for inference |
| `reports/metrics.json` | Full metrics for all trained models |
| `reports/model_comparison.csv` | Same data as CSV for easy sorting |
| `reports/roc_curves.png` | ROC curves for all models overlaid |
| `reports/shap_summary.png` | SHAP beeswarm feature importance plot |
| `reports/revenue_impact.csv` | Per-customer churn probability + revenue risk |

---

## 📡 Production Monitoring

Deploying a model is not the finish line. Here's how to keep it healthy in production.

### 1. Data Drift Monitoring

Track whether incoming feature distributions shift from training data over time.

**What to monitor:**
- `tenure`, `MonthlyCharges` — continuous distributions
- `Contract`, `InternetService` — category proportions

**Tool:** [Evidently AI](https://www.evidentlyai.com/) or a manual PSI (Population Stability Index) check.

**Threshold:**
- PSI > 0.10 → Warning
- PSI > 0.25 → Trigger retraining investigation

**Frequency:** Run weekly on a sample of recent predictions.

---

### 2. Model Performance Monitoring

Log every prediction. After 30 days, match predictions to actual churn outcomes.

**Metrics to track in production:**

| Metric | Alert Threshold |
|---|---|
| Rolling 30-day ROC-AUC | Alert if drops > 5% from baseline |
| Predicted churn rate vs actual | Alert if gap > 10 percentage points |
| Precision@K (top 200 flagged) | Alert if drops below 50% |

**Tool:** MLflow for experiment tracking + Grafana dashboards for live metrics.

---

### 3. Retraining Strategy

Use a **hybrid** approach:

```
Scheduled retraining (monthly)
  └── Retrain on rolling 12-month window
  └── Run champion vs challenger A/B test
  └── Promote only if new model beats baseline AUC

Event-driven retraining (triggered)
  └── Data drift alert fires (PSI > 0.25)
  └── Business event (price change, new product launch)
  └── AUC drops below 0.80 in production
```

**Model versioning:** Always tag and store the previous 2 model versions so rollback takes < 5 minutes.

---

### 4. Business KPI Monitoring

ML metrics don't tell the whole business story. Track:

| KPI | Target | How to Measure |
|---|---|---|
| Monthly churn rate | Decrease MoM | CRM data |
| Revenue saved from retained customers | Positive ROI | Retained churners × MRR |
| Intervention response rate | > 15% | % of at-risk customers accepting offer |
| False positive cost | Minimise | Discounts given to non-churners |
| Model coverage | > 95% | % of active customers scored |

---

## 🧰 Tech Stack

| Purpose | Library | Version |
|---|---|---|
| Data processing | `pandas` | ≥ 1.5 |
| Numerical ops | `numpy` | ≥ 1.23 |
| ML models | `scikit-learn` | ≥ 1.2 |
| Gradient boosting | `xgboost` | ≥ 1.7 |
| Hyperparameter tuning | `optuna` | ≥ 3.0 |
| Explainability | `shap` | ≥ 0.41 |
| Visualisation | `matplotlib`, `seaborn` | ≥ 3.6 |
| API serving | `fastapi` + `uvicorn` | ≥ 0.95 |
| Notebooks | `jupyter` | ≥ 1.0 |

---

## 🧪 Running Individual Modules

You don't have to run the full pipeline every time. Each module is independently executable:

```bash
# Just load and clean data
python src/data_loader.py

# Just run feature engineering
python src/feature_engineering.py

# Just retrain models (after preprocessing is done)
python src/train.py

# Just run evaluation (after models are trained)
python src/evaluate.py

# Just recompute revenue impact
python src/revenue_model.py

# Just rerun SHAP analysis
python src/shap_analysis.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- Dataset: [IBM Sample Data Sets](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
- SHAP library: [Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874)
- Optuna: [Akiba et al. (2019)](https://arxiv.org/abs/1907.10902)

---

<div align="center">

**Built for learning · Structured for production · Ready for interviews**

</div>