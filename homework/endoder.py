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

class MNIST_Autoencoder(nn.Module):
    def __init__(self, mode = 'training'):
        super(MNIST_Autoencoder, self).__init__()
        self.mode = mode
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True)
            )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
            )
        self.encoder_outputs = []
        
    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        if self.mode == 'test':
            self.encoder_outputs.append(encoder_out)
        return decoder_out

epochs = 10
train_batch_size, val_batch_size = 10, 2000
lr = 0.001

loss_list = []

train_loader, validation_loader = get_dataloader(train_batch_size, val_batch_size)

model = MNIST_Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

for epoch in trange(epochs):
    print(epoch)
    for step, (img, label) in enumerate(train_loader):
        img, label = img.view(-1,28*28).to(device), label.to(device)
        
        m_img = model(img)
        
        optimizer.zero_grad()
        loss = criterion(m_img, img)
        loss.backward()
        optimizer.step()

#%%
print(model.state_dict().keys())
trained_dict = model.state_dict()
from collections import OrderedDict

enc_state_dict = OrderedDict()
dec_state_dict = OrderedDict()

for k, v in trained_dict.items():
    if k.startswith('encoder'):
        enc_state_dict[k] = v

for k, v in trained_dict.items():
    if k.startswith('decoder'):
        dec_state_dict[k] = v

#%%
class MNIST_encoder(nn.Module):
    def __init__(self):
        super(MNIST_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU()
            )
    
    def forward(self, x):
        return self.encoder(x)
    
class MNIST_generator(nn.Module):
    def __init__(self):
        super(MNIST_generator, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        return self.decoder(x)
    
encoder = MNIST_encoder()
encoder.load_state_dict(enc_state_dict)

generator = MNIST_generator()
generator.load_state_dict(dec_state_dict)        
        
for step, (img, label) in enumerate(validation_loader):
    img = img.view(-1, 28*28)
    
    encoded_arr = encoder(img)
    m_img = generator(encoded_arr)
    
    
    t_img = m_img[0].view(28,28).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize = (30,30))
    ax.imshow(t_img, 'gray')
    print(encoded_arr.shape)
    
    break        
#%%
random_num = torch.tensor(np.random.uniform(0, 2, size = (1,256)), dtype = torch.float)
m_img = generator(random_num)
    
t_img = m_img[0].view(28,28).detach().cpu().numpy()
fig, ax = plt.subplots(figsize = (30,30))
ax.imshow(t_img, 'gray')
    
    
    
    
    
    