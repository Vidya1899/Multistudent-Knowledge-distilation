# importing All the required library 
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
import sys

'''
The Loss funtion for train Student using knowledge Distilation 
The Alpha and Temperature are the hyperparameters for knowledge distilation ,
Default values for T is taken as 7 and Alpha is 0.5
'''
def loss_kd(outputs, labels, teacher_outputs, temparature, alpha):
    T = temparature
    soft_target = F.softmax(teacher_outputs/T, dim=1)
    hard_target = labels
    out = outputs  ## this is the input to softmax
    logp = F.log_softmax(out/T, dim=1)
    loss_soft_target = -torch.mean(torch.sum(soft_target * logp, dim=1))
    loss_hard_target = nn.CrossEntropyLoss()(out, hard_target)
    loss = loss_soft_target * T * T + alpha * loss_hard_target
    return loss

'''
    The get_ouputs functions generate teachers output for all the inputs and returns the list containing ouputs 
    for each batch.
'''
def get_outputs(model, dataloader):
   '''
   Used to get the output of the teacher network
   '''
   outputs = []
   with torch.no_grad():
       for inputs, labels in dataloader:
           inputs_batch, _ = inputs.cuda(), labels.cuda()
           output_batch = model(inputs_batch).data.cpu().numpy()
           outputs.append(output_batch)
   return outputs
# Function to Train the model for 1 epoch
def train_kd(model,teacher_out, optimizer, loss_kd, dataloader, temparature, alpha):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for i,(images, labels) in enumerate(dataloader):
        inputs = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs_teacher = torch.from_numpy(teacher_out[i]).to(device)
        loss = loss_kd(outputs,labels,outputs_teacher,temparature, 
                        alpha)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
   
    epoch_loss = running_loss / len(trainset)
    epoch_acc = running_corrects.double() / len(trainset)
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, 
              epoch_acc))
#Function to evaluate the trained model.
def eval_kd(model,teacher_out, optimizer, loss_kd, dataloader, temparature, alpha):
   model.eval()
   running_loss = 0.0
   running_corrects = 0
   for i,(images, labels) in enumerate(dataloader):
      inputs = images.to(device)
      labels = labels.to(device)
      outputs = model(inputs)
      outputs_teacher = torch.from_numpy(teacher_out[i]).cuda()
      loss = loss_kd(outputs,labels,outputs_teacher,temparature, 
                     alpha)
      _, preds = torch.max(outputs, 1)
      running_loss += loss.item() * inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)
   epoch_loss = running_loss / len(valset)
   epoch_acc = running_corrects.double() / len(valset)
   print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss,epoch_acc))
   return epoch_acc

def train_and_evaluate_kd(model, teacher_model, optimizer, loss_kd, trainloader, valloader, temparature, alpha, num_epochs=25):
   teacher_model.eval()
   best_model_wts = copy.deepcopy(model.state_dict())
   outputs_teacher_train = get_outputs(teacher_model, trainloader)
   outputs_teacher_val = get_outputs(teacher_model, valloader)
   print("Teacher’s outputs are computed now starting the training process-")
   best_acc = 0.0
   for epoch in range(num_epochs):
      print('Epoch {}/{}'.format(epoch, num_epochs - 1))
      print('-' * 10)
      
      # Training the student with the soft labes as the outputs from the teacher and using the loss_kd function
      
      train_kd(model, outputs_teacher_train, 
               optim.Adam(model.parameters()),loss_kd,trainloader, 
               temparature, alpha)
     
      # Evaluating the student network
      epoch_acc_val = eval_kd(model, outputs_teacher_val, optim.Adam(model.parameters()), loss_kd, valloader, temparature, alpha)
      if epoch_acc_val > best_acc:
         best_acc = epoch_acc_val
         best_model_wts = copy.deepcopy(model.state_dict())
         print('Best val Acc: {:4f}'.format(best_acc))
         model.load_state_dict(best_model_wts)
   return model


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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True)
for param in resnet.parameters():
   param.requires_grad = False
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)
resnet = resnet.to(device)
resnet.to(device) 
resnet.load_state_dict(torch.load('teachertry.pth'))
print("Loaded The model ------>")

modelName = 'densenet.pt'
model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121').to(device)
for param in model.parameters():
   param.requires_grad = False
model =nn.Sequential(model,nn.Linear(1000,10),nn.Softmax())
model.to(device)
student = train_and_evaluate_kd(model,resnet,optim.Adam(model.parameters()),loss_kd,trainloader,valloader,7,0.5,10)
torch.save(student.load_state_dict(),modelName)


modelName = 'googlenet.pt'
student2=torch.hub.load('pytorch/vision:v0.6.0','googlenet',pretrained=True)
for param in student2.parameters():
   param.requires_grad = False
student2=nn.Sequential(student2,nn.Linear(1000,10),nn.Softmax())model.to(device)
student = train_and_evaluate_kd(student2,resnet,optim.Adam(model.parameters()),loss_kd,trainloader,valloader,7,0.5,10)
torch.save(student.load_state_dict(),modelName)

