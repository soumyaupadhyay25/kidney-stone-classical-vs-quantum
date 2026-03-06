import numpy as np
from sklearn.decomposition import PCA

# load features
features = np.load("quantum_features_reduced.npy")
labels = np.load("quantum_labels_reduced.npy")

# reduce dimension
pca = PCA(n_components=6)

reduced_features = pca.fit_transform(features)

# save reduced features
np.save("quantum_features_reduced.npy", reduced_features)
np.save("quantum_labels_reduced.npy", labels)

print("Dimension reduction complete")
print("New feature shape:", reduced_features.shape)