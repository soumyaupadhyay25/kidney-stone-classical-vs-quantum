import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report,confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_dataset = datasets.ImageFolder("dataset/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features,256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256,2)
)

model.load_state_dict(torch.load("kidney_model.pth"))
model = model.to(device)
model.eval()

all_preds=[]
all_labels=[]

with torch.no_grad():
    for images,labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs,1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print(confusion_matrix(all_labels,all_preds))
print(classification_report(all_labels,all_preds))
