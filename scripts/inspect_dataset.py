import pandas as pd

# Path to your CSV
path = "data/processed/Comprehensive_Banking_Database_10k.csv"

# Load
df = pd.read_csv(path)

# Quick overview
print("ROWS, COLUMNS:", df.shape)
print("\nCOLUMNS:\n", df.columns.tolist())
print("\nDATA TYPES:\n", df.dtypes.value_counts().to_string())
print("\nFIRST 5 ROWS:\n", df.head().to_string(index=False))
print("\nNULLS BY COLUMN:\n", df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))
