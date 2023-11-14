import torch
import torch.nn as nn

class MnistMLP(nn.Module):

    def __init__(self):
        super(MnistMLP, self).__init__()

        height = 28
        width = 28
        n_channels = 1

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(height * width * n_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

        self.layers.apply(init_weights)


    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.layers(x)