import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths (adjust if needed)
input_path = "data/processed/Comprehensive_Banking_Database_10k.csv"

# Load dataset
df = pd.read_csv(input_path)

print("===== BASIC INFO =====")
print("ROWS, COLUMNS:", df.shape)
print("\nCOLUMNS:", df.columns.tolist())
print("\nDATA TYPES:\n", df.dtypes)
print("\nNULL VALUES:\n", df.isnull().sum())

# --- FRAUD DISTRIBUTION ---
print("\n===== FRAUD DISTRIBUTION =====")
print(df['Anomaly'].value_counts())
print(df['Anomaly'].value_counts(normalize=True))

plt.figure(figsize=(5, 4))
sns.countplot(x='Anomaly', data=df, palette="coolwarm")
plt.title("Fraud vs Non-Fraud Transactions")
plt.savefig("data/processed/fraud_distribution.png")
plt.close()

# --- TRANSACTION AMOUNT ---
plt.figure(figsize=(6, 4))
sns.histplot(df['Transaction Amount'], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.savefig("data/processed/transaction_amount.png")
plt.close()

# --- CORRELATION HEATMAP ---
plt.figure(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (numeric features)")
plt.savefig("data/processed/correlation_heatmap.png")
plt.close()

print("\nEDA complete! Plots saved in data/processed/")
