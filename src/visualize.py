import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.mlp import MLPNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
net = MLPNet()
net.load_state_dict(t.load("mnist_mlp.pth"))
net.eval()

# Show 10 sample predictions
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
with t.no_grad():
    for i in range(10):
        img, label = test_set[i]
        pred = net(img.unsqueeze(0)).argmax().item()
        axs[i//5, i%5].imshow(img.view(28, 28), cmap='gray')
        axs[i//5, i%5].set_title(f"Pred: {pred}, True: {label}")
        axs[i//5, i%5].axis('off')
plt.tight_layout()
plt.show()