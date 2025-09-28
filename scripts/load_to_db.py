import pandas as pd
import sqlite3

# Paths
input_path = "data/processed/Comprehensive_Banking_Database_10k_cleaned.csv"
db_path = "data/processed/banking.db"

# Load dataset
df = pd.read_csv(input_path)

# Connect to SQLite (creates file if not exists)
conn = sqlite3.connect(db_path)

# Write to SQL table
df.to_sql("banking_data", conn, if_exists="replace", index=False)

# Quick check: row count
rows = conn.execute("SELECT COUNT(*) FROM banking_data").fetchone()[0]
print(f"✅ Loaded dataset into SQLite: {rows} rows")

conn.close()
