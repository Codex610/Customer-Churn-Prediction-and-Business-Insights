import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from app_config import REPORTS_DIR

TREE_MODELS = (RandomForestClassifier, XGBClassifier, LGBMClassifier)


def run_shap(model, X_test, model_name="best_model"):
    """Generate SHAP values using the correct explainer for the model type."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"[shap] Model type: {type(model).__name__}")

    # ── Pick the right explainer ──────────────────────────
    if isinstance(model, TREE_MODELS):
        print("[shap] Using TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)

        # Binary classification returns list [class0, class1] — take class 1
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    else:
        # LogisticRegression and any other linear/non-tree model
        print("[shap] Using LinearExplainer...")
        explainer = shap.LinearExplainer(model, X_test)
        shap_vals = explainer.shap_values(X_test)

    # ── Ensure clean 2D float64 array (n_samples, n_features) ──
    shap_vals = np.array(shap_vals, dtype=np.float64)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)

    X_test_np     = np.array(X_test, dtype=np.float64)
    feature_names = list(X_test.columns)

    # ── Summary bar plot ──────────────────────────────────
    plt.figure()
    shap.summary_plot(
        shap_vals,
        X_test_np,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, "shap_summary.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[shap] SHAP summary saved to {save_path}")

    # ── Top 10 features by mean |SHAP| ───────────────────
    mean_shap = pd.DataFrame({
        "feature":   feature_names,
        "mean_shap": np.abs(shap_vals).mean(axis=0)
    }).sort_values("mean_shap", ascending=False)

    print("\n[shap] Top 10 features:")
    print(mean_shap.head(10).to_string(index=False))

    return shap_vals