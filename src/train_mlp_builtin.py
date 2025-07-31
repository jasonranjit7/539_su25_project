import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from models.mlp import MLPNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

net = MLPNet()
loss_fn = CrossEntropyLoss()
optimizer = t.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for ep in range(10):
    net.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        output = net(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {ep+1}, Loss: {total_loss:.4f}")

t.save(net.state_dict(), 'results/mlp_builtin.pth')