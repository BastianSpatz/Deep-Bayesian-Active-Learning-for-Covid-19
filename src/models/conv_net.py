import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv3d layer input shape:
# N -> number of sequences (mini batch)
# Cin -> number of channels (3 for rgb)
# D -> Number of images in a sequence
# H -> Height of one image in the sequence
# W -> Width of one image in the sequence

class ConvNN(nn.Module):
    
    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 512,
        img_cols: int = 512,
        maxpool: int = 2,
    ):
        """
        Basic Architecture of CNN
        Attributes:
            num_filters: Number of filters, out channel for 1st and 2nd conv layers,
            kernel_size: Kernel size of convolution,
            dense_layer: Dense layer units,
            img_rows: Height of input image,
            img_cols: Width of input image,
            maxpool: Max pooling size
        """
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 3, (3, 3, 3), 1, bias=False)
        self.conv2 = nn.Conv3d(3, 3, (3, 3, 3), 1, bias=False)

        self.batchnorm1 = nn.BatchNorm3d(3)
        self.batchnorm2 = nn.BatchNorm3d(3)

        self.avgpool = nn.AdaptiveAvgPool3d(8)# 256 x 1 x 1

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(3*8*8*8, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)
        x = self.dropout1(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        out = self.fc2(x)
        return out