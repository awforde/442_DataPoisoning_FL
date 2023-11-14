import torch
import torch.nn as nn

class Cifar10MLP(nn.Module):

    def __init__(self):
        super(Cifar10MLP, self).__init__()

        height = 32
        width = 32
        n_channels = 3

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(height * width * n_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.layers(x)