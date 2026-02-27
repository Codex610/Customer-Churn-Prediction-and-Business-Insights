"""
main.py — End-to-end pipeline runner
Usage: python main.py
"""

from data_loader         import load_data
from feature_engineering import engineer_features
from preprocessing       import preprocess
from train               import train_all, save_best_model
from evaluate            import evaluate_all, plot_confusion_matrix, plot_roc_curve, pick_best_model
from tune                import get_tuned_model
from revenue_model       import estimate_revenue_impact
from shap_analysis       import run_shap
from config              import TARGET


def run_pipeline(tune=False):
    print("\n" + "="*50)
    print("  Telco Customer Churn — ML Pipeline")
    print("="*50 + "\n")

    # ── 1. Load ───────────────────────────────────────────
    print("── Step 1: Load Data")
    df = load_data()

    # ── 2. Feature Engineering ────────────────────────────
    print("\n── Step 2: Feature Engineering")
    df = engineer_features(df)

    # ── 3. Preprocessing ──────────────────────────────────
    print("\n── Step 3: Preprocessing")
    df, scaler = preprocess(df)

    # ── 4. Train Base Models ──────────────────────────────
    print("\n── Step 4: Train Models")
    results, X_train, X_test, y_train, y_test = train_all(df)

    # ── 5. Evaluate ───────────────────────────────────────
    print("\n── Step 5: Evaluate Models")
    all_metrics, comparison_df = evaluate_all(results)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    plot_roc_curve(results)
    best_name, best_model = pick_best_model(all_metrics, results)

    # ── 6. (Optional) Tune Best Model ─────────────────────
    if tune:
        print(f"\n── Step 6: Tune {best_name}")
        best_model = get_tuned_model(X_train, y_train, model_name=best_name)
        # Re-wrap for evaluate helpers
        results[best_name + "_tuned"] = {
            "model": best_model, "X_test": X_test, "y_test": y_test
        }

    # ── 7. Save Best Model ────────────────────────────────
    print("\n── Step 7: Save Best Model")
    save_best_model(best_model)
    plot_confusion_matrix(best_model, X_test, y_test, model_name=best_name)

    # ── 8. Revenue Impact ─────────────────────────────────
    print("\n── Step 8: Revenue Impact")
    estimate_revenue_impact(df, X_test, y_test, best_model)

    # ── 9. SHAP Analysis ──────────────────────────────────
    print("\n── Step 9: SHAP Analysis")
    run_shap(best_model, X_test, model_name=best_name)

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    # Set tune=True to run Optuna hyperparameter tuning (slower)
    run_pipeline(tune=False)
