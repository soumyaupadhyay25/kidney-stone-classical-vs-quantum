import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score

# Ground truth labels
y_true = np.load("quantum_labels_reduced.npy")

# Example predictions (replace with real predictions later)
classical_pred = np.random.randint(0,2,len(y_true))
quantum_sim_pred = np.random.randint(0,2,len(y_true))
quantum_ibm_pred = np.random.randint(0,2,len(y_true))

# -------- Time values (example realistic values) --------
classical_time = 0.03
quantum_sim_time = 0.01
quantum_ibm_time = 23

# -------- Metrics --------

def get_metrics(y_true,y_pred):

    acc = accuracy_score(y_true,y_pred)
    prec = precision_score(y_true,y_pred)

    return acc,prec


classical_acc,classical_prec = get_metrics(y_true,classical_pred)
sim_acc,sim_prec = get_metrics(y_true,quantum_sim_pred)
ibm_acc,ibm_prec = get_metrics(y_true,quantum_ibm_pred)

models = ["Classical","Quantum Simulator","Quantum IBM"]

accuracy = [classical_acc,sim_acc,ibm_acc]
precision = [classical_prec,sim_prec,ibm_prec]
time_taken = [classical_time,quantum_sim_time,quantum_ibm_time]

# -------- Accuracy Graph --------

plt.figure()

plt.bar(models,accuracy)

plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()

# -------- Precision Graph --------

plt.figure()

plt.bar(models,precision)

plt.title("Precision Comparison")
plt.ylabel("Precision")

plt.show()

# -------- Time Graph --------

plt.figure()

plt.bar(models,time_taken)

plt.title("Execution Time Comparison")
plt.ylabel("Seconds")

plt.show()