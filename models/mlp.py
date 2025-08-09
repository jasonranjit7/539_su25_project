import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dims=(100, 50), num_classes=10, dropout_p=0.0):
        super().__init__()
        h1, h2 = hidden_dims
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
