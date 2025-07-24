import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

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

# Neural network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(28*28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x

net = Net()
loss_fn = nn.CrossEntropyLoss()
lr = 0.01
momentum = 0.9

# Manual SGD with momentum
def sgd(net, lr, momentum=0.9):
    with t.no_grad():
        for p in net.parameters():
            if p.grad is None:
                continue
            if not hasattr(p, 'velocity'):
                p.velocity = t.zeros_like(p.grad)
            p.velocity = momentum * p.velocity + (1 - momentum) * p.grad
            p -= lr * p.velocity
            p.grad = None  # Clear the gradient

# Training loop
epochs = 10
for ep in range(epochs):
    net.train()
    total_loss = []
    for x, y in train_loader:
        output = net(x)
        loss = loss_fn(output, y)
        loss.backward()
        sgd(net, lr, momentum)
        total_loss.append(loss.detach().item())
    print(f"Epoch {ep+1}, Loss: {loss:.4f}")

# Evaluation
correct = 0
total = 0
net.eval()
with t.no_grad():
    for x, y in test_loader:
        output = net(x)
        preds = t.argmax(output, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f'Accuracy: {round(correct / total, 3)}')

# Visualize a sample prediction
sample_img = x[3]
print(f'Predicted label: {t.argmax(net(sample_img.view(-1, 784))[0])}')
plt.imshow(sample_img.view(28, 28), cmap='gray')
plt.show()
