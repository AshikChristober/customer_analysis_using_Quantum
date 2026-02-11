import matplotlib.pyplot as plt

# Read accuracy values
with open("../results/accuracy.txt", "r") as f:
    lines = f.readlines()

classical_acc = float(lines[0].split(":")[1])
quantum_acc = float(lines[1].split(":")[1])

models = ["Classical ML", "Quantum ML"]
scores = [classical_acc, quantum_acc]

plt.bar(models, scores)
plt.ylabel("Accuracy")
plt.title("Classical vs Quantum Customer Behaviour Analysis")

# Save plot
plt.savefig("../results/comparison_plot.png")
plt.show()

print("📈 Comparison plot saved as comparison_plot.png")

