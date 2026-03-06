import numpy as np
import time

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# -----------------------------
# Load reduced features
# -----------------------------

features = np.load("../quantum_features_reduced.npy")
labels = np.load("../quantum_labels_reduced.npy")

# Quantum hardware slow hota hai
# Isliye sirf 50 samples run karenge

features = features[:50]
labels = labels[:50]

print("Loaded samples:", len(features))


# -----------------------------
# Connect to IBM Quantum
# -----------------------------

service = QiskitRuntimeService()

backend = service.backend("ibm_fez")

sampler = Sampler(backend)

print("Connected to backend:", backend)


# -----------------------------
# Build quantum circuits
# -----------------------------

circuits = []

for x in features:

    qc = QuantumCircuit(2)

    qc.ry(float(x[0]), 0)
    qc.ry(float(x[1]), 1)

    qc.cx(0, 1)

    qc.measure_all()

    qc = transpile(qc, backend)

    circuits.append(qc)


print("Total circuits:", len(circuits))


# -----------------------------
# Run on IBM Quantum hardware
# -----------------------------

start = time.time()

job = sampler.run(circuits)

print("IBM JOB ID:", job.job_id())

result = job.result()

end = time.time()


print("Execution finished")
print("Total time:", round(end - start, 3), "seconds")
print("Circuits executed:", len(circuits))