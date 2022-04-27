from turtle import forward
from typing import OrderedDict
import tvm
from tvm import topi
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.modules
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,5,1,2)
        self.conv2 = nn.Conv2d(32,64,5,1)
        self.fc1 = nn.Linear(1600,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x
        


#create network
net = LeNet()
#create loss function
loss = torch.nn.CrossEntropyLoss()

#create SGD solver and attach network and loss
solver = torch.optim.SGD(net.parameters(),0.01)


import torch
import torch.utils
import torchvision

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=32, shuffle=True)


import time
curT = time.time()

# Apply one epoch
for batch_idx, (data, target) in enumerate(train_loader):
    solver.zero_grad()
    x = net(data)
    y = loss(x,target)
    y.backward()
    solver.step()

    answers = x.detach().numpy().argmax(1)
    acc = (answers==target.numpy()).sum()/data.shape[0]
    curSpeed = (time.time()-curT)/(batch_idx+1)
    print("Torch:{} | accuracy:{} | second per iteration:{}".format(y.item(),acc,curSpeed))



