import pandas as pd
df = pd.read_csv("data/equipment_data.csv")
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("\nMaintenance Required - unique values and dtype:")
print(df["Maintenance Required"].dtype)
print(df["Maintenance Required"].unique()[:30])
print("\nSample rows:")
print(df.head(10).to_string())
