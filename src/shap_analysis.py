import os
import shap
import matplotlib.pyplot as plt

from config import REPORTS_DIR


def run_shap(model, X_test, model_name="best_model"):
    """Generate SHAP values and save beeswarm summary plot."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("[shap] Computing SHAP values...")
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    # Beeswarm summary plot
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f"SHAP Summary — {model_name}")
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, "shap_summary.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"[shap] SHAP summary saved to {save_path}")

    # Top 10 important features by mean |SHAP|
    import pandas as pd
    mean_shap = pd.DataFrame({
        "feature":    X_test.columns,
        "mean_shap":  abs(shap_values.values).mean(axis=0)
    }).sort_values("mean_shap", ascending=False)

    print("\n[shap] Top 10 features:")
    print(mean_shap.head(10).to_string(index=False))

    return shap_values
