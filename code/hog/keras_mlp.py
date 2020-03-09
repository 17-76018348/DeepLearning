from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


#%%
from tqdm import trange
import matplotlib.pyplot as plt



import torch
import torch.nn as nn
import torch.optim  as optim


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


x_train = torch.tensor(x_train, dtype = torch.float).view(-1, 28 * 28)
y_train = torch.tensor(y_train, dtype = torch.long).view(-1)
x_test = torch.tensor(x_test, dtype = torch.float).view(-1, 28 * 28)
y_test = torch.tensor(y_test, dtype = torch.float).view(-1)





class Hog_MLP(nn.Module):
    def __init__(self, p):
        super(Hog_MLP, self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim = -1)
        
        
        
        
        
        
    def forward(self, x):
        print("\n")
        print(x.shape)
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
        x4 = self.relu(x3)
        x5 = self.fc3(x4)
        print(x5.shape)
        x6 = self.softmax(x5)
        return x6



epochs = 30

lr = 0.001
cnt = 0
loss_list = []


model = Hog_MLP(p = 0).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr = lr)

for epoch in trange(epochs):
        model.train()
        loss_epoch = 0
        for step, img in enumerate(x_train):
            
            img = img.view(-1,28*28).to(device)
            label = y_train[step].to(device)
            pred = model(img)
            optimizer.zero_grad()
            print("\n")
            print(img.shape)
            print(pred.shape)
            print(label.shape)
            loss = criterion(pred,label)
            loss_epoch += loss.item() * pred.shape[0]
            loss.backward()
            optimizer.step()
        loss_epoch /= len(x_train)
        loss_list.append(loss_epoch)

        print(epoch, loss_epoch)
        
    
    
fig, ax = plt.subplots(1, 1, figsize = (30, 15))
ax.plot(loss_list)