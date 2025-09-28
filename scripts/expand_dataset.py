import pandas as pd
import numpy as np
from faker import Faker

# Paths
in_path = "data/raw/Comprehensive_Banking_Database.csv"
out_path = "data/processed/Comprehensive_Banking_Database_10k.csv"

# Init faker
fake = Faker()

# Load original
df = pd.read_csv(in_path)

# Duplicate
df2 = df.copy()

# Offset IDs so they don’t clash
df2["Customer ID"] = df2["Customer ID"] + df["Customer ID"].max()

# Modify a few columns randomly
np.random.seed(42)
df2["Age"] = df2["Age"].apply(lambda x: max(18, x + np.random.randint(-5, 6)))
df2["Account Balance"] = df2["Account Balance"] * np.random.uniform(0.8, 1.2, size=len(df2))
df2["Transaction Amount"] = df2["Transaction Amount"] * np.random.uniform(0.7, 1.3, size=len(df2))
df2["Loan Amount"] = df2["Loan Amount"] * np.random.uniform(0.8, 1.2, size=len(df2))

# Regenerate names/emails
df2["First Name"] = [fake.first_name() for _ in range(len(df2))]
df2["Last Name"] = [fake.last_name() for _ in range(len(df2))]
df2["Email"] = [fake.email() for _ in range(len(df2))]
df2["Contact Number"] = [fake.msisdn() for _ in range(len(df2))]
df2["Address"] = [fake.address().replace("\n", ", ") for _ in range(len(df2))]

# Merge original + new
df_final = pd.concat([df, df2], ignore_index=True)

# Save
df_final.to_csv(out_path, index=False)

print("Expanded dataset saved to:", out_path)
print("New size:", df_final.shape)
