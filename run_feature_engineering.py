import sys
sys.path.insert(0, "src")

from data_loader import load_data
from feature_engineering import engineer_features

df = load_data()
df = engineer_features(df)
df.to_csv("data/processed/features_temp.csv", index=False)
print("Feature engineering done.")
