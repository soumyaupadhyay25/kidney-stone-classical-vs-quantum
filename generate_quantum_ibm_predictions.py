import numpy as np
import time

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

features = np.load("quantum_features_reduced.npy")

service = QiskitRuntimeService()

backend = service.least_busy(simulator=False, operational=True)

sampler = Sampler(backend)

predictions = []

start = time.time()

# run only small subset on hardware
for x in features[:20]:

    qc = QuantumCircuit(2)

    qc.rx(x[0],0)
    qc.rx(x[1],1)

    qc.cx(0,1)

    qc.measure_all()

    qc = transpile(qc,backend)

    job = sampler.run([qc])

    result = job.result()

    counts = result[0].data.meas.get_counts()

    # prediction from measurement
    if "11" in counts:
        pred = 1
    else:
        pred = 0

    predictions.append(pred)

end = time.time()

hardware_time = end - start

predictions = np.array(predictions)

np.save("quantum_ibm_predictions.npy",predictions)

print("IBM hardware predictions saved")
print("Execution time:",hardware_time)
print("Backend used:",backend.name)