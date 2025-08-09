import torch
import torch.nn as nn

class SimpleCNN(nn.Module):

    def __init__(self, in_channels=1, num_classes=10, dropout_p=0.0):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self._flatten_dim = None

        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.size(0)
        x = self.feature(x)

        if self._flatten_dim is None:
            self._flatten_dim = x.shape[1] * x.shape[2] * x.shape[3]
            if self._flatten_dim != self.fc1.in_features:
                self.fc1 = nn.Linear(self._flatten_dim, 64)

        x = x.view(bs, -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x