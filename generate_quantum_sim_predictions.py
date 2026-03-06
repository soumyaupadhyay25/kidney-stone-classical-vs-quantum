import numpy as np

features = np.load("quantum_features_reduced.npy")

predictions = []

for f in features:

    value = np.mean(f)

    if value > 0.5:
        predictions.append(1)
    else:
        predictions.append(0)

predictions = np.array(predictions)

np.save("quantum_sim_predictions.npy",predictions)

print("Quantum simulator predictions generated")
print("Samples:",len(predictions))