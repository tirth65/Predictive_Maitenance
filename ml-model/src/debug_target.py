# src/debug_target.py
import pandas as pd
from src.preprocess import preprocess_data
from src.features import add_features
import os

csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "equipment_data.csv"))
print("RAW CSV path:", csv)
df_raw = pd.read_csv(csv)
print("RAW columns:", df_raw.columns.tolist())
print("Sample raw head:\n", df_raw.head(3).to_string())

# Check if original target exists
print("\nRaw has 'Maintenance Required'?:", "Maintenance Required" in df_raw.columns)
if "Maintenance Required" in df_raw.columns:
    print("Maintenance Required dtype and unique:", df_raw["Maintenance Required"].dtype, df_raw["Maintenance Required"].unique()[:10])

# Run your preprocess + feature pipeline and inspect columns
df_p = preprocess_data(df_raw.copy())
print("\nColumns AFTER preprocess_data():", df_p.columns.tolist())
df_f = add_features(df_p.copy(), include_trend=False)
print("\nColumns AFTER add_features():", df_f.columns.tolist())

# If target lost, print a short message
print("\nDoes processed data keep 'Maintenance Required'?:", "Maintenance Required" in df_f.columns)
# If not, list candidate columns containing 'maint' substring (case-insensitive)
cands = [c for c in df_f.columns if 'maint' in c.lower() or 'maintenance' in c.lower()]
print("Candidate columns in processed dataframe (contain 'maint'):", cands)
