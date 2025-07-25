import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models.mlp import MLPNet

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = t.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# Model
net = MLPNet()
net.load_state_dict(t.load('mnist_mlp.pth'))
net.eval()

# Evaluation
correct = 0
total = 0
all_preds = []
all_labels = []

with t.no_grad():
    for x, y in test_loader:
        output = net(x)
        preds = output.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(all_labels, all_preds))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.show()