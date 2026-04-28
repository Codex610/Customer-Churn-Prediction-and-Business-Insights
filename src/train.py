import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

import dagshub
dagshub.init(repo_owner='codex03080', repo_name='Customer-Churn-Prediction-and-Business-Insights', mlflow=True)

from config import (TARGET, TEST_SIZE, RANDOM_STATE, MODEL_PATH, MODELS,
                    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)


def get_X_y(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "xgboost":             XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0),
        "lightgbm":            LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    }


def train_all(df):
    """Train all models, log each run to MLflow, return results dict."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models  = get_models()
    results = {}

    for name, model in models.items():
        if name not in MODELS:
            continue

        with mlflow.start_run(run_name=name):
            # Log hyperparameters
            mlflow.log_param("model_type",   name)
            mlflow.log_param("test_size",    TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_params(model.get_params())

            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, artifact_path=name)

            results[name] = {
                "model":  model,
                "X_test": X_test,
                "y_test": y_test,
            }
            print(f"[train] ✅ {name} trained and logged to MLflow.")

    return results, X_train, X_test, y_train, y_test


def save_best_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"[train] Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import engineer_features
    from preprocessing import preprocess

    df = load_data()
    df = engineer_features(df)
    df, _ = preprocess(df)
    results, *_ = train_all(df)
