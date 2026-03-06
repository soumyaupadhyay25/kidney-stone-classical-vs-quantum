import matplotlib.pyplot as plt

models = ["Classical CNN", "Quantum Simulator", "IBM Quantum Hardware"]

# seconds
times = [12, 5, 527]

plt.figure(figsize=(8,5))

plt.bar(models, times)

plt.title("Execution Time Comparison")
plt.ylabel("Time (seconds)")
plt.xlabel("Model")

for i,v in enumerate(times):
    plt.text(i, v + 5, str(v), ha='center')

plt.tight_layout()

plt.savefig("time_comparison.png")

plt.show()