import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.mlp import MLPNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=10, shuffle=False)

net = MLPNet()
net.load_state_dict(t.load('results/mlp_manual.pth'))
net.eval()

correct = 0
total = 0
all_preds = []
all_labels = []

with t.no_grad():
    for x, y in test_loader:
        output = net(x)
        preds = t.argmax(output, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

print(f'Test Accuracy: {correct / total:.4f}')

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")

sample = x[0]
plt.imshow(sample.view(28, 28), cmap='gray')
plt.title(f"Predicted: {preds[0].item()}")
plt.show()