import pandas as pd

# Paths
input_path = "data/processed/Comprehensive_Banking_Database_10k.csv"
output_path = "data/processed/Comprehensive_Banking_Database_10k_cleaned.csv"

# Load dataset
df = pd.read_csv(input_path)

# --- Cleaning steps ---
# 1. Standardize column names
df.columns = (
    df.columns.str.strip()       # remove extra spaces
             .str.lower()        # make lowercase
             .str.replace(" ", "_")  # spaces → underscores
             .str.replace("/", "_")  # slashes → underscores
)

# 2. Convert date columns
date_cols = [
    "date_of_account_opening",
    "last_transaction_date",
    "transaction_date",
    "approval_rejection_date",
    "payment_due_date",
    "last_credit_card_payment_date",
    "feedback_date",
    "resolution_date"
]
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# 3. Convert numeric columns (safe)
num_cols = [
    "account_balance",
    "transaction_amount",
    "account_balance_after_transaction",
    "loan_amount",
    "interest_rate",
    "loan_term",
    "credit_limit",
    "credit_card_balance",
    "minimum_payment_due",
    "rewards_points"
]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Save cleaned dataset
df.to_csv(output_path, index=False)

print(f"✅ Cleaned dataset saved to {output_path}")
print("ROWS, COLUMNS:", df.shape)
print("NULLS after cleaning:", df.isnull().sum().sum())
