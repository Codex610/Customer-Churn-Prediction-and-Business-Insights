import os
import json
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from config import REPORTS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def compute_metrics(model, X_test, y_test):
    """Return a dict of classification metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }


def evaluate_all(results):
    """Evaluate all trained models, log metrics to MLflow, save reports."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_metrics = {}

    for name, res in results.items():
        metrics = compute_metrics(res["model"], res["X_test"], res["y_test"])
        all_metrics[name] = metrics

        # Log metrics to the existing run for this model
        with mlflow.start_run(run_name=f"{name}_eval"):
            mlflow.log_metrics(metrics)

        print(f"[evaluate] {name}: {metrics}")

    # Save metrics.json
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)

    # Save model_comparison.csv
    comparison_df = pd.DataFrame(all_metrics).T.reset_index()
    comparison_df.rename(columns={"index": "model"}, inplace=True)
    comparison_df.sort_values("roc_auc", ascending=False, inplace=True)
    comparison_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    # Log comparison CSV as MLflow artifact
    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_artifact(os.path.join(REPORTS_DIR, "model_comparison.csv"))

    print(f"[evaluate] Reports saved to {REPORTS_DIR}")
    return all_metrics, comparison_df


def plot_confusion_matrix(model, X_test, y_test, model_name="best_model"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    save_path = os.path.join(REPORTS_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(save_path)
    plt.show()

    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name=f"{model_name}_confusion_matrix"):
        mlflow.log_artifact(save_path)
    print(f"[evaluate] Confusion matrix saved and logged.")


def plot_roc_curve(results):
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        y_proba = res["model"].predict_proba(res["X_test"])[:, 1]
        fpr, tpr, _ = roc_curve(res["y_test"], y_proba)
        auc = roc_auc_score(res["y_test"], y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(REPORTS_DIR, "roc_curves.png")
    plt.savefig(save_path)
    plt.show()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name="roc_curves"):
        mlflow.log_artifact(save_path)
    print("[evaluate] ROC curve saved and logged.")


def pick_best_model(all_metrics, results):
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["roc_auc"])
    print(f"[evaluate] Best model: {best_name} (AUC={all_metrics[best_name]['roc_auc']})")
    return best_name, results[best_name]["model"]
