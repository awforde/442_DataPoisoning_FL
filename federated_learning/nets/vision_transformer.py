import torch
import torch.nn as nn
from torchvision.models import vision_transformer

class CustomVisionTransformer(nn.Module):

    def __init__(self):
        super(CustomVisionTransformer, self).__init__()


        ds = "CIFAR-10"
        # ds = "MNIST"


        # Start with the default ViT model
        self.model = vision_transformer.vit_b_16(weights='DEFAULT')

        embedding_dim = 768

        # Modify the first convolutional layer to work with the dataset
        # CIFAR-10 images are 3x32x32; the default patch size and channels are good
        if ds == "MNIST":
            # MNIST images are 1x28x28
            # Replace the first layer so the patch size and input channels work
            self.model.conv_proj = nn.Conv2d(1, embedding_dim, kernel_size=(14,14), stride=(14,14))
        else:
            print("ERROR! Invalid dataset type detected!")


        # Modify the model's fully connected layer to match 10 classes
        self.model.heads.head = nn.Linear(embedding_dim, 10)


    def forward(self, x):
        return self.model(x)
