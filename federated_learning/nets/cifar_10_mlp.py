import torch
import torch.nn as nn

class Cifar10MLP(nn.Module):

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self):
        super(Cifar10MLP, self).__init__()

        height = 32
        width = 32
        n_channels = 3

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(height * width * n_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

        # Initialize the weights of the MLP
        self.layers.apply(init_weights)



    def forward(self, x):
        return self.layers(x)