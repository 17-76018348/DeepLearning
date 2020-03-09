import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
import sys
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




class MNIST_MLP(nn.Module):
    def __init__(self, p):
        super(MNIST_MLP, self).__init__()
        self.model = nn.Sequential(
                nn.Dropout(p = p),
                nn.Linear(28*28,128),
                nn.ReLU(),
                nn.Dropout(p = p),
                nn.Linear(128,64),
                nn.ReLU(),

                nn.Linear(64,10),
                nn.LogSoftmax(dim = 1)
            )
        
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    
    

epochs = 3
trian_batch_size, val_batch_size = 3, 2000
lr = 0.001
cnt = 0
loss_list = []
val_acc_list = []
train_loader, validation_loader = get_dataloader(trian_batch_size,val_batch_size)


model = MNIST_MLP(p = 0.3).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr = lr)
    
    
    
for epoch in trange(epochs):
    model.train()
    loss_epoch = 0
    for step, (img, label) in enumerate(train_loader):
        

        img, label = img.view(-1,28*28).to(device), label.to(device)

        pred = model(img)

        optimizer.zero_grad()
        print("\n")

        print(label)

        sys.exit(0)
        loss = criterion(pred,label)

        loss_epoch += loss.item() * pred.shape[0]
        loss.backward()
        optimizer.step()

    loss_epoch /= len(train_loader.dataset)
    loss_list.append(loss_epoch)
    
    model.eval()
    val_acc = 0
    
    for step, (img, label) in enumerate(validation_loader):
        img, label = img.view(-1,28*28).to(device), label.to(device)
        
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