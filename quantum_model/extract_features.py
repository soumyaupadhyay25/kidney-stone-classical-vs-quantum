import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import numpy as np

# dataset path
data_dir = "dataset"

# image preprocessing
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=16)

# load pretrained CNN
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# remove classifier layer
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()

features = []
labels = []

with torch.no_grad():
    for imgs, lbls in loader:

        out = feature_extractor(imgs)

        out = out.view(out.size(0), -1)

        features.append(out.numpy())
        labels.append(lbls.numpy())

features = np.vstack(features)
labels = np.hstack(labels)

# save features
np.save("quantum_features.npy", features)
np.save("quantum_labels.npy", labels)

print("Feature extraction complete")
print("Feature shape:", features.shape)