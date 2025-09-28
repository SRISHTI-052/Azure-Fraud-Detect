# scripts/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# ------------------------------
# Paths
# ------------------------------
train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"
model_path = "models/fraud_rf_model.pkl"

# ------------------------------
# Load train/test
# ------------------------------
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop("anomaly", axis=1)
y_train = train_df["anomaly"]

X_test = test_df.drop("anomaly", axis=1)
y_test = test_df["anomaly"]

print(f"✅ Data loaded: Train {X_train.shape}, Test {X_test.shape}")

# Ensure y_train contains discrete labels
print("Unique values in y_train before mapping:", y_train.unique())  # Debugging step

# Map continuous or unexpected values to discrete classes (if necessary)
if y_train.dtype != 'int' and y_train.dtype != 'bool':
    y_train = y_train.map(lambda x: 1 if x > 0 else 0)  # Adjust mapping logic as needed

print("Unique values in y_train after mapping:", y_train.unique())  # Debugging step

# ------------------------------
# Handle imbalance with SMOTE
# ------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("✅ Resampled train data:", X_train_res.shape)

# ------------------------------
# Train Random Forest
# ------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train_res, y_train_res)
print("✅ Model trained!")

# ------------------------------
# Evaluate
# ------------------------------
y_pred = rf.predict(X_test)  # Ensure discrete class predictions
y_proba = rf.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

# Debugging step: Check unique values in y_test and y_pred
print("Unique values in y_test:", y_test.unique())
print("Unique values in y_pred:", pd.Series(y_pred).unique())

print("===== Classification Report =====")
print(classification_report(y_test, y_pred))

print("===== Confusion Matrix =====")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# ------------------------------
# Save model
# ------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(rf, model_path)
print(f"✅ Model saved to {model_path}")
