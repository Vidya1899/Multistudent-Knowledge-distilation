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

class teacher(nn.Module):
    def __init__(self):
        super(teacher,self).__init__()
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.num_ftrs, 10)
    def forward(self,x):
        x = self.resnet(x)
        return x

class densenet(nn.Module):
    def __init__(self):
        super(densenet,self).__init__()
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dense = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121' )
        self.linear = nn.Linear(1000,10)
    def forward(self,x):
        x = self.dense(x)
        x = self.linear(x)
        return F.softmax(x)

class googlenet(nn.Module):
    def __init__(self):
        super(googlenet,self).__init__()
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dense = torch.hub.load('pytorch/vision:v0.6.0','googlenet')
        self.linear = nn.Linear(1000,10)
    def forward(self,x):
        x = self.dense(x)
        x = self.linear(x)
        return F.softmax(x)

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