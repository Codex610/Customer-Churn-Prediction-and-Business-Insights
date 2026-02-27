import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from config import NUM_COLS, CAT_COLS, TARGET, DATA_PROCESSED, SCALER_PATH


def handle_missing(df):
    """Fill missing values."""
    # TotalCharges has a few nulls (new customers with no charges yet)
    df["TotalCharges"].fillna(0, inplace=True)
    return df


def encode_categoricals(df):
    """Binary encode Yes/No columns and one-hot encode multi-class columns."""
    # Columns with Yes / No → 1 / 0
    binary_cols = [
        col for col in CAT_COLS
        if df[col].dropna().isin(["Yes", "No"]).all()
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Encode target
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    # One-hot encode remaining categorical columns
    remaining_cats = [col for col in CAT_COLS if col not in binary_cols]
    df = pd.get_dummies(df, columns=remaining_cats, drop_first=True)

    return df


def scale_numericals(df, fit=True, scaler=None):
    """Scale numerical columns. If fit=True, fit a new scaler and save it."""
    if fit:
        scaler = StandardScaler()
        df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])
        joblib.dump(scaler, SCALER_PATH)
        print(f"[preprocessing] Scaler saved to {SCALER_PATH}")
    else:
        df[NUM_COLS] = scaler.transform(df[NUM_COLS])
    return df, scaler


def preprocess(df, fit=True, scaler=None):
    """Full preprocessing pipeline."""
    df = handle_missing(df)
    df = encode_categoricals(df)
    df, scaler = scale_numericals(df, fit=fit, scaler=scaler)

    # Save processed data
    df.to_csv(DATA_PROCESSED, index=False)
    print(f"[preprocessing] Processed data saved to {DATA_PROCESSED}")

    return df, scaler


if __name__ == "__main__":
    from data_loader import load_data
    raw = load_data()
    clean, scaler = preprocess(raw)
    print(clean.head())
