import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from models.mlp import MLPNet
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10, shuffle=False)

net = MLPNet()
loss_fn = CrossEntropyLoss()
lr = 0.01
momentum = 0.9

def manual_sgd(net, lr, momentum):
    with t.no_grad():
        for p in net.parameters():
            if p.grad is None:
                continue
            if not hasattr(p, 'velocity'):
                p.velocity = t.zeros_like(p.grad)
            p.velocity = momentum * p.velocity + (1 - momentum) * p.grad
            p -= lr * p.velocity
            p.grad = None

train_losses = []
train_accuracies = []

for ep in range(10):
    net.train()
    correct = 0
    total = 0
    epoch_loss = 0

    for x, y in train_loader:
        output = net(x)
        loss = loss_fn(output, y)
        loss.backward()
        manual_sgd(net, lr, momentum)
        epoch_loss += loss.item()

        preds = t.argmax(output, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(acc)
    print(f"Epoch {ep+1}: Loss={epoch_loss:.4f}, Accuracy={acc:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Accuracy')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("results/accuracy_plots.png")

t.save(net.state_dict(), 'results/mlp_manual.pth')