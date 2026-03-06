import numpy as np
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA

from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler

# ----------------------------
# Load Data
# ----------------------------
X = np.load("quantum_features.npy")
y = np.load("quantum_labels.npy")

print("Loaded data:", X.shape)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# ----------------------------
# Connect to IBM Quantum
# ----------------------------
service = QiskitRuntimeService()

backend = service.least_busy(simulator=False)
print("Running on backend:", backend.name)

# ✅ NEW correct session creation (no 'service=' argument)
with Session(backend=backend) as session:

    sampler = Sampler(session=session)

    # ----------------------------
    # Build Quantum Model
    # ----------------------------
    feature_map = ZZFeatureMap(feature_dimension=6, reps=2)
    ansatz = RealAmplitudes(6, reps=2)

    optimizer = COBYLA(maxiter=60)

    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
    )

    print("Training Quantum Model on IBM Hardware...")

    vqc.fit(X, y)

    pred = vqc.predict(X)
    accuracy = (pred == y).mean()

    print("Quantum Accuracy:", accuracy)