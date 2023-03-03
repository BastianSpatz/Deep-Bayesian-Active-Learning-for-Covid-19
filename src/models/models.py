import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, mobilenet_v2, MobileNet_V2_Weights
from torch.hub import load_state_dict_from_url
import baal.bayesian.dropout


class MobileNetV2(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=False)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.model.features[0][0] = nn.Conv2d(
            64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.model.classifier = nn.Sequential(
            baal.bayesian.dropout.Dropout(p=0.25),
            nn.Linear(in_features=1280, out_features=512),
            nn.ReLU(),
            # baal.bayesian.dropout.Dropout(p=0.2),
            # nn.Linear(in_features=1000, out_features=512),
            # nn.ReLU(),
            baal.bayesian.dropout.Dropout(p=0.25),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            baal.bayesian.dropout.Dropout(p=0.25),
            nn.Linear(in_features=256, out_features=3),
        )
        print(self.model)

    def forward(self, x):
        return self.model(x)


class CustomVGG16(nn.Module):
    def __init__(self, active_learning_mode=False) -> None:
        super(CustomVGG16, self).__init__()
        self.model = vgg16(pretrained=True)
        # weights = load_state_dict_from_url(
        #     "https://download.pytorch.org/models/vgg16-397923af.pth"
        # )
        # weights = {k: v for k, v in weights.items() if "classifier.6" not in k}
        # self.model.load_state_dict(weights, strict=False)

        self.model.features = nn.Sequential(*list(self.model.children())[0][2:])
        self.model.avgpool = nn.Sequential(nn.Identity())
        for param in self.model.parameters():
            param.requires_grad = False

        if active_learning_mode:
            self.dropout = baal.bayesian.dropout.Dropout(p=0.2)
        else:
            self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Flatten(),
            nn.Linear(in_features=16384, out_features=4096),
            nn.ReLU(),
            self.dropout,
            nn.Linear(in_features=4096, out_features=256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(in_features=256, out_features=3),
        )
        self.model = nn.Sequential(
            self.model.features, self.model.avgpool, self.classifier
        )
        print(self.model)

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )  # fully connected layer, output 10 classes
        self.out = nn.Linear(524288, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output  # return x for visualization


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
        self.conv1 = self._conv_layer_set(64, 128)
        self.conv2 = self._conv_layer_set(128, 128)
        self.conv3 = self._conv_layer_set(128, 64)
        self.fc1 = nn.Linear(2097152, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 3)
        self.relu = nn.ReLU()
        # self.skip = nn.Identity()
        # self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.fc1_bn = nn.BatchNorm1d(128)
        self.cov1_bn = nn.BatchNorm3d(8)
        self.cov2_bn = nn.BatchNorm3d(16)
        self.cov3_bn = nn.BatchNorm3d(32)
        self.drop = nn.Dropout(p=0.25)
        # self.avgpool = nn.AdaptiveAvgPool3d(16)# 256 x 1 x 1

    def _conv_layer_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        return conv_layer

    def forward(self, x):
        # print('input shape:', x.shape)
        x = self.conv1(x)
        # x = self.cov1_bn(x)
        # x = self.max_pool(x)

        x = self.conv2(x)
        # x = self.cov2_bn(x)
        # x = self.max_pool(x)

        x = self.conv3(x)
        # x = self.cov3_bn(x)
        # x = self.max_pool(x)
        # x = self.conv3_bn(x)
        # print('before flatten shape:', x.shape)

        # print('after flatten shape:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc1_bn(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
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

        self.avgpool = nn.AdaptiveAvgPool3d(8)  # 256 x 1 x 1

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(3 * 8 * 8 * 8, 128)
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
