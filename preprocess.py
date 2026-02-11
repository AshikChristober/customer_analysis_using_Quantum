import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load raw data
df = pd.read_csv("/home/ashik-christober/Desktop/Quantum_Neueral_Network/data/WA_Fn-UseC_-Telco-Customer-Churn.xls")

# Drop customer ID (not useful for learning)
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing numeric values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Save cleaned dataset
df.to_csv("../data/cleaned_data.csv", index=False)

print("✅ Preprocessing complete: cleaned_data.csv saved")

