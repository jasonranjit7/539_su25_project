import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from models.mlp import MLPNet

# Transform: normalize input
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = t.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)

test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = t.utils.data.DataLoader(test_set, batch_size=10, shuffle=False)

# Model
net = MLPNet()
loss_fn = nn.CrossEntropyLoss()
lr = 0.01
momentum = 0.9

# Manual SGD
def sgd(net, lr, momentum=0.9):
    with t.no_grad():
        for p in net.parameters():
            if p.grad is None:
                continue
            if not hasattr(p, 'velocity'):
                p.velocity = t.zeros_like(p.grad)
            p.velocity = momentum * p.velocity + (1 - momentum) * p.grad
            p -= lr * p.velocity
            p.grad = None

# Training
epochs = 10
for ep in range(epochs):
    net.train()
    total_loss = []
    for x, y in train_loader:
        output = net(x)
        loss = loss_fn(output, y)
        loss.backward()
        sgd(net, lr, momentum)
        total_loss.append(loss.item())
    print(f"Epoch {ep+1}, Loss: {loss:.4f}")

# Save model
t.save(net.state_dict(), 'mnist_mlp.pth')