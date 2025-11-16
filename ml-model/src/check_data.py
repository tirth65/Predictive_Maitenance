# save as src/check_data.py and run: python -m src.check_data
import pandas as pd
df = pd.read_csv("data/equipment_data.csv")
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("\nfailure value counts:")
print(df['failure'].value_counts(dropna=False))
print("\nfirst 5 rows:")
print(df.head().to_string())
# If you have machine id / timestamp:
for col in ['machine_id','id','timestamp','time','unit']:
    if col in df.columns:
        print(f"found column: {col} (sample values: {df[col].unique()[:5]})")
