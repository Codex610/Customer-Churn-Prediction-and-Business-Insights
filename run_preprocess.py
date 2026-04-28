import sys
sys.path.insert(0, "src")

import pandas as pd
from preprocessing import preprocess

df = pd.read_csv("data/processed/features_temp.csv")
clean, _ = preprocess(df)
print("Preprocessing done.")
