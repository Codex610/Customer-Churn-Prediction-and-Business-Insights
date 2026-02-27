import pandas as pd
from config import TARGET


def add_tenure_buckets(df):
    """Bucket tenure into 4 groups."""
    # Work on original tenure before scaling — if already scaled, skip
    # This should be called BEFORE preprocessing/scaling
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "12-24m", "24-48m", "48-72m"]
    )
    # One-hot encode the new bucket column
    df = pd.get_dummies(df, columns=["tenure_group"], drop_first=True)
    return df


def add_interaction_features(df):
    """Create interaction features based on EDA insights."""

    # High monthly charge flag (above median ~$65)
    df["high_monthly_charges"] = (df["MonthlyCharges"] > 65).astype(int)

    # Charges per month of tenure (spend rate)
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    return df


def add_service_count(df):
    """Count how many add-on services a customer has."""
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # These are still raw Yes/No strings at this point
    existing = [c for c in service_cols if c in df.columns]
    df["service_count"] = df[existing].apply(
        lambda row: sum(row == "Yes"), axis=1
    )
    return df


def engineer_features(df):
    """Run all feature engineering steps. Call BEFORE preprocessing."""
    df = add_tenure_buckets(df)
    df = add_service_count(df)
    df = add_interaction_features(df)
    print(f"[feature_engineering] Done. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    df = engineer_features(df)
    print(df.head())
