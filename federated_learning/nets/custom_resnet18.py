import torch
import torch.nn as nn
from torchvision.models import resnet18


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()

        ds = "CIFAR-10"
        # ds = "MNIST"

        # Start with the pre-trained resnet18 model
        self.resnet18 = resnet18(weights="DEFAULT")

        # Modify the first convolutional layer to work with the dataset
        if ds == "CIFAR-10":
            # CIFAR-10 images are 3x32x32, while ImageNet images are 3x224x224
            self.resnet18.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
        elif ds == "MNIST":
            # MNIST images are 1x28x28
            self.resnet18.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        else:
            print("ERROR! Invalid dataset type detected!")

        # Initialize the first layer's weights
        nn.init.xavier_uniform_(self.resnet18.conv1.weight)

        # Modify the model's fully connected layer to match 10 classes
        self.resnet18.fc = nn.Linear(512, 10)
        # Initialize the fc layer weights
        nn.init.xavier_uniform_(self.resnet18.fc.weight)
        self.resnet18.fc.bias.data.fill_(0.01)

    def forward(self, x):
        return self.resnet18(x)
