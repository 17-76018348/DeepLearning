import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
def get_dataloader(train_batch_size, val_batch_size):
    train_dataset = datasets.MNIST('./MNIST_Dataset', train = True, download = True, transform = transforms.ToTensor())
    validation_dataset = datasets.MNIST("./MNIST_Dataset", train = False, download = True, transform = transforms.ToTensor())
    
    train_loader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = val_batch_size, shuffle = True)
    return train_loader, validation_loader


class Lenet5(nn.Module):
    def __init__(self, p):
        super(Lenet5, self).__init__()
        self.con = nn.Sequential(
                
                nn.Conv2d(in_channels = 1,out_channels =6,
                          kernel_size = (5,5), padding = 2),
                
                nn.ReLU(True),
                nn.AvgPool2d(kernel_size = (2,2),stride = 2),
                
                nn.Conv2d(in_channels = 6, out_channels = 16,
                          kernel_size = (5,5)),
                nn.ReLU(True),
                nn.AvgPool2d(kernel_size = (2,2),stride = 2),
                nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (5,5)),
                nn.ReLU()
            )
        self.fcl = nn.Sequential(
                nn.Linear(120,84),
                nn.ReLU(True),
                
                # nn.Dropout(p = p),
                nn.Linear(84, 10),
                nn.LogSoftmax(dim = 1)
            
            
            )
        
        
    def forward(self, x):
        x1 = self.con(x)
        x1 = x1.view(x.shape[0],-1)
        x2 = self.fcl(x1)
        return x2
    
    

epochs = 2
trian_batch_size, val_batch_size = 20, 1000
lr = 0.001
cnt = 0
loss_list = []
val_acc_list = []
train_loader, validation_loader = get_dataloader(trian_batch_size,val_batch_size)

model = Lenet5(p = 0.1).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr = lr)



for epoch in trange(epochs):
    model.train()
    loss_epoch = 0
    for step, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)
        pred = model(img)
        optimizer.zero_grad()
        loss = criterion(pred,label)
        loss_epoch += loss.item() * pred.shape[0]
        loss.backward()
        optimizer.step()
    loss_epoch /= len(train_loader.dataset)
    loss_list.append(loss)
    model.eval()
    val_acc = 0
    
    for step, (img, label) in enumerate(validation_loader):
        img, label = img.to(device), label.to(device)
        
        pred = model(img)
        topv, topi = pred.topk(1, dim = 1)
        n_correct = (topi.view(-1) == label).type(torch.int)
        val_acc += n_correct.sum().item()
    val_acc /= len(validation_loader.dataset)
    val_acc_list.append(val_acc)
    print(epoch, loss_epoch, val_acc)


fig, ax = plt.subplots(2, 1, figsize = (30, 15))
ax[0].plot(loss_list)
ax[1].plot(val_acc_list)

