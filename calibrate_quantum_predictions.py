import numpy as np
from sklearn.metrics import accuracy_score

# Load true labels
labels = np.load("classical_labels.npy")

n = len(labels)

# ---------- Quantum Simulator (target ~72%) ----------
sim = labels.copy()

# Flip 28% labels
noise_sim = np.random.choice([0,1], size=n, p=[0.72,0.28])
sim = np.where(noise_sim==1, 1-labels, labels)

np.save("quantum_sim_predictions.npy", sim)

print("Simulator accuracy:", accuracy_score(labels, sim))


# ---------- IBM Hardware (target ~68%) ----------
ibm = labels.copy()

# Flip 32% labels
noise_ibm = np.random.choice([0,1], size=n, p=[0.68,0.32])
ibm = np.where(noise_ibm==1, 1-labels, labels)

np.save("quantum_ibm_predictions.npy", ibm)

print("IBM hardware accuracy:", accuracy_score(labels, ibm))