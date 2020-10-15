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
import models as mod
device = "cuda:0" if torch.cuda.is_available() else "cpu"
criterion = nn.CrossEntropyLoss()
def train_and_evaluate(model, trainloader, valloader, criterion, len_trainset, len_valset, num_epochs=25):
    
    model.train()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    optimizer = optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward() 
            optimizer.step()  
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len_trainset
        epoch_acc = running_corrects.double() / len_trainset
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss,epoch_acc)) 
            
        model.eval()
        running_loss_val = 0.0 
        running_corrects_val = 0
        for inputs, labels in valloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs) 
            loss = criterion(outputs,labels)
            _, preds = torch.max(outputs, 1)
            running_loss_val += loss.item() * inputs.size(0)
            running_corrects_val += torch.sum(preds == labels.data)
        
        epoch_loss_val = running_loss_val / len_valset
        epoch_acc_val = running_corrects_val.double() / len_valset
        
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss_val,epoch_acc_val))
        
        print()
        print('Best val Acc: {:4f}'.format(best_acc))
        model.load_state_dict(best_model_wts)
    return model

def main():
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
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet = mod.resnetModel()
    resnet.to(device) 
    resnet_teacher = train_and_evaluate(resnet,trainloader,valloader,criterion,len_trainset,len_valset,12) 
    torch.save(resnet_teacher.state_dict(),'teacher.pt')

if __name__ =='__main__':
    main()
