import pandas as pd
from config import DATA_RAW


def load_data(path=DATA_RAW):
    """Load raw Telco churn CSV and return a DataFrame."""
    df = pd.read_csv(path)

    # Fix TotalCharges — loaded as string due to whitespace values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customerID — not a feature
    df.drop(columns=["customerID"], inplace=True)

    print(f"[data_loader] Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
