import torch
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features,256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256,2)
)

model.load_state_dict(torch.load("classical_model/kidney_model.pth", map_location="cpu"))
model.eval()

for name,param in model.named_parameters():
    print(name, torch.mean(param).item())
    break