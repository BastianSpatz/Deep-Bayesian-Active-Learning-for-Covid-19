import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv3d layer input shape:
# N -> number of sequences (mini batch)
# Cin -> number of channels (3 for rgb)
# D -> Number of images in a sequence
# H -> Height of one image in the sequence
# W -> Width of one image in the sequence

class MnistExampleModel(nn.Module):
    
    def __init__(self):
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
        super(MnistExampleModel, self).__init__()
        self.conv1 = self._conv_layer_set(3, 3)
        self.conv2 = self._conv_layer_set(3, 3)
        self.conv3 = self._conv_layer_set(3, 3)
        self.fc1 = nn.Linear(3*8*8*8, 128)
        self.fc2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()
        self.skip = nn.Identity()
        self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.fc1_bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.25)
        self.avgpool = nn.AdaptiveAvgPool3d(8)# 256 x 1 x 1

    def _conv_layer_set(self, in_channels: int, out_channels: int) -> torch.nn.Sequential:
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=(3, 3, 3), 
                stride=1,
                padding=1,
                ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            )
        return conv_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('input shape:', x.shape)
        x = self.conv1(x)
        x = self.max_pool(x)


        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.max_pool(x)
        # x = self.conv3_bn(x)
        # print('before flatten shape:', x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print('after flatten shape:', x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc1_bn(x)
        x = self.drop(x)
        x = self.fc2(x)
        # print('output shape:', x.shape)

        return x

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