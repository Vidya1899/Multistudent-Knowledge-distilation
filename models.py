import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch import nn, optim
from collections import  OrderedDict

def resnetModel():
    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
       param.requires_grad = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 10)
    return resnet

def densenet():
    modelName = 'densenet.pt'
    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121')
    for param in model.parameters():
       param.requires_grad = False
    model =nn.Sequential(model,nn.Linear(1000,10),nn.Softmax())
    return model

def googlenet():
    student2=torch.hub.load('pytorch/vision:v0.6.0','googlenet',pretrained=True)
    for param in student2.parameters():
       param.requires_grad = False
    student2=nn.Sequential(student2,nn.Linear(1000,10),nn.Softmax())
    return student2

class CNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4704, 1),
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = F.sigmoid(self.fc_layer(x))

        return x
