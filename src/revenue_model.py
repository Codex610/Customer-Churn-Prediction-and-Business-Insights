import os
import pandas as pd

from app_config import REPORTS_DIR, AVG_CUSTOMER_LIFETIME


def estimate_revenue_impact(df_original, X_test, y_test, model):
    """
    Estimate revenue at risk for each customer in the test set.

    Revenue at risk = churn probability × MonthlyCharges × avg lifetime months
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    # Build output dataframe
    revenue_df = X_test.copy()
    revenue_df["churn_probability"]  = y_proba
    revenue_df["actual_churn"]       = y_test.values

    # Use actual MonthlyCharges if available, else fallback to config default
    if "MonthlyCharges" in revenue_df.columns:
        revenue_df["revenue_at_risk"] = (
            revenue_df["churn_probability"] *
            revenue_df["MonthlyCharges"] *
            AVG_CUSTOMER_LIFETIME
        )
    else:
        from app_config import AVG_MONTHLY_REVENUE
        revenue_df["revenue_at_risk"] = (
            revenue_df["churn_probability"] *
            AVG_MONTHLY_REVENUE *
            AVG_CUSTOMER_LIFETIME
        )

    revenue_df.sort_values("revenue_at_risk", ascending=False, inplace=True)

    # Save report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "revenue_impact.csv")
    revenue_df[["churn_probability", "actual_churn", "revenue_at_risk"]].to_csv(out_path)

    total_at_risk = revenue_df["revenue_at_risk"].sum()
    print(f"[revenue_model] Total revenue at risk: ${total_at_risk:,.2f}")
    print(f"[revenue_model] Report saved to {out_path}")

    return revenue_df
