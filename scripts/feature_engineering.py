import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Paths
input_path = "data/processed/Comprehensive_Banking_Database_10k_cleaned.csv"
output_train_path = "data/processed/train.csv"
output_test_path = "data/processed/test.csv"

# Load dataset
df = pd.read_csv(input_path)
print("✅ Loaded dataset:", df.shape)

# ------------------------------
# 1. Encode categorical columns
# ------------------------------
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}")

# ------------------------------
# 2. Scale numeric columns
# ------------------------------
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop("fraud", errors='ignore')
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("Scaled numeric columns")

# ------------------------------
# 3. Train/Test Split
# ------------------------------
if "anomaly" in df.columns:
    X = df.drop("anomaly", axis=1)
    y = df["anomaly"]
else:
    raise ValueError("❌ 'anomaly' column not found in dataset. Check cleaned data.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed train/test
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

os.makedirs("data/processed", exist_ok=True)
train_df.to_csv(output_train_path, index=False)
test_df.to_csv(output_test_path, index=False)

print(f"✅ Train saved: {output_train_path} ({train_df.shape})")
print(f"✅ Test saved: {output_test_path} ({test_df.shape})")
