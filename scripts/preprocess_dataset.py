import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Paths
input_path = "data/processed/Comprehensive_Banking_Database_10k.csv"
output_train = "data/processed/train.csv"
output_test = "data/processed/test.csv"

# Load dataset
df = pd.read_csv(input_path)

# ----- 1. Drop irrelevant columns -----
drop_cols = [
    "Customer ID", "First Name", "Last Name", "Address", "City", "Contact Number",
    "Email", "TransactionID", "Loan ID", "CardID", "Feedback ID"
]
df = df.drop(columns=drop_cols)

# ----- 2. Handle dates -----
# Convert date columns to datetime
date_cols = [
    "Date Of Account Opening", "Last Transaction Date", "Transaction Date",
    "Approval/Rejection Date", "Payment Due Date", "Last Credit Card Payment Date",
    "Feedback Date", "Resolution Date"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Create new numeric features (example: account age in days, transaction weekday)
df["Account_Age_Days"] = (pd.to_datetime("today") - df["Date Of Account Opening"]).dt.days
df["Transaction_Weekday"] = df["Transaction Date"].dt.weekday

# Drop original date columns (too complex for ML directly)
df = df.drop(columns=date_cols)

# ----- 3. Separate features (X) and target (y) -----
X = df.drop(columns=["Anomaly"])
y = df["Anomaly"].map({1: 0, -1: 1})  # 0 = normal, 1 = fraud

# ----- 4. Identify categorical and numeric columns -----
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ----- 5. Build preprocessing pipeline -----
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Pipeline = preprocessing only (no model yet)
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Fit + transform
X_processed = pipeline.fit_transform(X)

# Convert back to DataFrame
# Get feature names
ohe_features = pipeline.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_cols)
all_features = numeric_cols + list(ohe_features)
X_processed_df = pd.DataFrame(X_processed, columns=all_features)  # Remove .toarray()

# Add target back
final_df = X_processed_df.copy()
final_df["Fraud"] = y.values

# ----- 6. Train-test split -----
train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42, stratify=final_df["Fraud"])

# Save outputs
os.makedirs("data/processed", exist_ok=True)
train_df.to_csv(output_train, index=False)
test_df.to_csv(output_test, index=False)

print("✅ Preprocessing complete!")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"Saved to: {output_train}, {output_test}")
