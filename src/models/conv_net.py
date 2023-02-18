import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.resnet import resnet10 
from torchvision.models import vgg16
import baal.bayesian.dropout 

class CustomVGG16(nn.Module):
    def __init__(self, active_learning_mode=False) -> None:
        super(CustomVGG16, self).__init__()
        self.model = vgg16(pretrained=True)
        
        self.model.features = nn.Sequential(*list(self.model.children())[0][2:])
        self.model.avgpool = nn.Sequential(nn.Identity())
        for param in self.model.parameters():
            param.requires_grad = False
        # Add on classifier
        # self.model.classifier[6] = nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 3))
        if active_learning_mode:
            self.dropout = baal.bayesian.dropout.Dropout(p=0.5)
        else:
            self.dropout = nn.Dropout(0.5)
        # num_features = self.model.classifier[6].in_features
        # features = list(self.model.classifier.children())[:-1] # Remove last layer
        # features.extend([nn.Linear(num_features, 3)]) # Add our layer with 4 outputs
        # self.classifier = baal.bayesian.dropout.patch_module(nn.Sequential(*features)) # Replace the model classifier
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
        self.model = nn.Sequential(self.model.features, self.classifier)
        print(self.model)

    def forward(self, x):
        # x = self.model.features(x)
        # # print(x.shape) 

        # x = self.classifier(x)
        return self.model(x)

class MedicalNet(nn.Module):

  def __init__(self, path_to_weights, device):
    super(MedicalNet, self).__init__()
    self.model = resnet10(sample_input_D=1, sample_input_H=112, sample_input_W=112, num_seg_classes=3)
    self.model.conv_seg = nn.Sequential(
        nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
        nn.Flatten(start_dim=1),
        nn.Dropout(0.25)
    )
    net_dict = self.model.state_dict()
    pretrained_weights = torch.load(path_to_weights, map_location=torch.device(device))
    pretrain_dict = {
        k.replace("module.", ""): v for k, v in pretrained_weights['state_dict'].items() if k.replace("module.", "") in net_dict.keys()
      }
    net_dict.update(pretrain_dict)
    self.model.load_state_dict(net_dict)
    for param_name, param in self.model.named_parameters():
        if param_name.startswith("conv_seg") or param_name.startswith("conv"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    self.fc = nn.Linear(512, 3)


  def forward(self, x):
    features = self.model(x)
    return self.fc(features)
  
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=64,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )        # fully connected layer, output 10 classes
        self.out = nn.Linear(524288, 3)    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output   # return x for visualization

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