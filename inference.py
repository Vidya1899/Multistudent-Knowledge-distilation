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
import math
import torch
from torch.autograd import Variable
import models as mod

transform = transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,  
                            0.406], [0.229, 0.224, 0.225])])
trainset = datasets.CIFAR10('train/', download=True, train=True, transform=transform)
valset = datasets.CIFAR10('val/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
len_trainset = len(trainset)
len_valset = len(valset)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




student1 = mod.densenet()
student1.to(device)
student1.load_state_dict(torch.load('densenetV2.pth'))

student2 = mod.googlenet()
student2.to(device)
student2.load_state_dict(torch.load('googlenetV2.pth'))

selector = mod.CNN()
selector.to(device)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for inputs, labels in valloader:
        inputs = inputs.to(device)
        model = torch.tensor(selector(inputs).flatten() >0.5,dtype=torch.uint8)
        _,s1 = torch.max(student1(inputs),1)
        _,s2 = torch.max(student2(inputs),1)
        for j in range(len(model)):
            if model[j] ==0:
                s1[j] = s2[j]
        predicted = s1
        
        c = (predicted == labels.cuda()).squeeze()
        for i in range(len(model)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

print("Accuracy of model" , sum(class_correct)/sum(class_total))
