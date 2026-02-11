import pennylane as qml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load cleaned data
df = pd.read_csv("../data/cleaned_data.csv")

X = df.drop("Churn", axis=1).values
y = df["Churn"].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Number of qubits
n_qubits = 4
X = X[:, :n_qubits]  # Reduce dimensions to match qubits

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Quantum device
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Initialize weights
weights = np.random.randn(2, n_qubits)

def quantum_features(X):
    return np.array([quantum_circuit(x, weights) for x in X])

# Extract quantum features
X_train_q = quantum_features(X_train)
X_test_q = quantum_features(X_test)

# Train classifier on quantum features
q_model = LogisticRegression(max_iter=1000)
q_model.fit(X_train_q, y_train)

# Prediction
y_pred_q = q_model.predict(X_test_q)
q_accuracy = accuracy_score(y_test, y_pred_q)

print(f"⚛️ Quantum ML Accuracy: {q_accuracy:.4f}")

# Append result
with open("../results/accuracy.txt", "a") as f:
    f.write(f"Quantum ML Accuracy: {q_accuracy:.4f}\n")

