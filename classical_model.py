import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load cleaned data
df = pd.read_csv("../data/cleaned_data.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classical ML model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"📊 Classical ML Accuracy: {accuracy:.4f}")

# Save result
with open("../results/accuracy.txt", "w") as f:
    f.write(f"Classical ML Accuracy: {accuracy:.4f}\n")

