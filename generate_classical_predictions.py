import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# Load model
model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features,256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256,2)
)

model.load_state_dict(torch.load("classical_model/kidney_model.pth", map_location="cpu"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

dataset_path = "dataset/val"

predictions = []
labels = []

for label in ["normal","stone"]:

    folder = os.path.join(dataset_path,label)

    for img in sorted(os.listdir(folder)):

        if img.startswith("."):
            continue

        path = os.path.join(folder,img)

        image = Image.open(path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():

            output = model(img_tensor)
            pred = torch.argmax(output,1).item()

        predictions.append(pred)

        if label == "stone":
            labels.append(1)
        else:
            labels.append(0)

predictions = np.array(predictions)
labels = np.array(labels)

np.save("classical_predictions.npy",predictions)
np.save("classical_labels.npy",labels)

print("Classical predictions generated")
print("Samples:",len(predictions))