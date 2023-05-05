from collections import OrderedDict
import sys
from baal.bayesian.dropout import patch_module
import torch
from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    vgg16,
    VGG16_Weights,
    VGG19_Weights,
    vgg19,
    vit_b_16,
    ViT_B_16_Weights,
    densenet121,
    DenseNet121_Weights,
)


class CustomVIT(nn.Module):
    def __init__(self, active_learning_mode=False, p=0.5) -> None:
        super(CustomVIT, self).__init__()
        self.model = vit_b_16(image_size=512, dropout=p)
        self.model.conv_proj = nn.Conv2d(64, 768, kernel_size=(16, 16), stride=(16, 16))
        self.model.heads = nn.Sequential(
            nn.Linear(in_features=768, out_features=3, bias=True),
        )
        if active_learning_mode:
            # change dropout layer to MCDropout
            self.model = patch_module(self.model)

        print(self.model)

    def forward(self, x):
        return self.model(x)


class CustomDenseNet(nn.Module):
    def __init__(self, active_learning_mode=False, p=0.5) -> None:
        super(CustomDenseNet, self).__init__()
        self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1, drop_rate=p)

        print(self.model)
        sys.exit()
        if active_learning_mode:
            # change dropout layer to MCDropout
            self.model = patch_module(self.model)

        print(self.model)

    def forward(self, x):
        return self.model(x)


class CustomVGG19(nn.Module):
    def __init__(self, active_learning_mode=False, p=0.5) -> None:
        super(CustomVGG19, self).__init__()
        self.model = vgg16(weights=VGG16_Weights.DEFAULT)

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p, inplace=False),
            nn.Linear(in_features=1024, out_features=3, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=p, inplace=False),
            # nn.Linear(in_features=256, out_features=128, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=p, inplace=False),
            # nn.Linear(in_features=128, out_features=3, bias=True)
        )
        # self.model.features = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=False),
        #     *list(self.model.children())[0][2:]
        # )

        if active_learning_mode:
            # change dropout layer to MCDropout
            self.model = patch_module(self.model)

        print(self.model)

    def forward(self, x):
        return self.model(x)


class Conv3DModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Conv3DModel, self).__init__(*args, **kwargs)

        self.model = nn.Sequential(
            # features
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(5, 5, 5)),
            nn.BatchNorm3d(64),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(5, 5, 5)),
            nn.BatchNorm3d(128),
            # classifier
            nn.Flatten(),
            nn.Linear(90112, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 3),
        )

    def forward(self, x):
        return self.model(x)
