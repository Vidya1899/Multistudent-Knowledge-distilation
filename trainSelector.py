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
def onehot(lab):
    y_onehot = torch.FloatTensor(lab.shape[0], 10)
    y_onehot.zero_()
    for i,j in enumerate(lab):
        y_onehot[i][j]=1
    return y_onehot

def createPrior(student1,student2,trainloader):
    classes = []
    criterion = nn.CrossEntropyLoss()
    for i ,(inputs,label) in enumerate(trainloader):
        with torch.no_grad():
            inputs = inputs.cuda()
            label = onehot(label).cuda()
            s1 = student1(inputs)
            s2 = student2(inputs)
            d1 = torch.sum(torch.abs(s1-label),dim = 1)
            d2 = torch.sum(torch.abs(s2-label),dim = 1)
            classes.append((d1<d2).type(torch.uint8))
    torch.save(classes,'prior.pth')

def trainSelector(sel,trainloader):
    classes = torch.load('prior.pth')
    print(classes[0])
    optimizer = optim.Adam(sel.parameters(),lr =0.001)
    cri = nn.BCELoss()

    for epoch in range(20):
        step=0
        for i ,(inputs,label) in enumerate(trainloader):
            optimizer.zero_grad()
            y = sel(inputs.cuda()).flatten()
            loss = cri(torch.tensor(y,requires_grad=True).cpu(),torch.tensor(classes[step],dtype= torch.float32))
            step+=1
            loss.backward()
            optimizer.step()
            if i%150 ==0:
                print('epoch {}, step {}, loss:{}'.format(epoch,i,loss.item()))
    torch.save(sel.load_state_dict(),'selector.pth')

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,  
                                0.406], [0.229, 0.224, 0.225])])
trainset = datasets.CIFAR10('train/', download=True, train=True, transform=transform)
valset = datasets.CIFAR10('val/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64)
valloader = torch.utils.data.DataLoader(valset, batch_size=64)
len_trainset = len(trainset)
len_valset = len(valset)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


student1 = mod.densenet()
student1.to(device)
student1.load_state_dict(torch.load('densenetV2.pth'))

student2= mod.googlenet()
student2.to(device)
student2.load_state_dict(torch.load('googlenetV2.pth'))

selector = mod.CNN()
selector.to(device)
createPrior(student1,student2,trainloader)
trainSelector(selector,trainloader)
