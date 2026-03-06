import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score

labels = np.load("classical_labels.npy")

classical = np.load("classical_predictions.npy")
sim = np.load("quantum_sim_predictions.npy")
ibm = np.load("quantum_ibm_predictions.npy")

# Match lengths
sim = sim[:len(labels)]
ibm = ibm[:len(labels)]

# If IBM has fewer samples
min_len = min(len(labels), len(ibm))
labels_ibm = labels[:min_len]
ibm = ibm[:min_len]

print("\nCLASSICAL MODEL")
print("Accuracy:", accuracy_score(labels,classical))
print("Precision:", precision_score(labels,classical))
print("F1:", f1_score(labels,classical))

print("\nQUANTUM SIMULATOR")
print("Accuracy:", accuracy_score(labels,sim))
print("Precision:", precision_score(labels,sim))
print("F1:", f1_score(labels,sim))

print("\nIBM QUANTUM HARDWARE")
print("Accuracy:", accuracy_score(labels_ibm,ibm))
print("Precision:", precision_score(labels_ibm,ibm))
print("F1:", f1_score(labels_ibm,ibm))