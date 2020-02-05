##### Check Overfitting for various n_neuron
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

from utils import dataset_generator, tester

np.random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

x_train_data, y_train_data, x_test_data, y_test_data = dataset_generator()


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (30,15))
# cmap = cm.get_cmap('bwr_r', 2)
# ax1.grid()
# ax2.grid()
# ax1.set_title("Train Dataset", fontsize = 30)
# ax2.set_title("Test Dataset", fontsize = 30)
# fig.subplots_adjust(top = 0.9, bottom = 0.1, left = 0.1, right = 0.9,
#                     wspace = 0.05)
# ax1.scatter(x_train_data[:,0], x_train_data[:,1], marker = 'o', color = cmap(y_train_data), alpha = 0.4)
# ax2.scatter(x_test_data[:,0], x_test_data[:,1], marker = 'o', color = cmap(y_test_data), alpha = 0.4)

class MLP_Classifier(nn.Module):
    def __init__(self, n_neuron):
        super(MLP_Classifier, self).__init__()
        self.n_neuron = n_neuron
        self.fc1 = nn.Linear(2, n_neuron)
        self.fc2 = nn.Linear(n_neuron, 1)
        # self.fc3 = nn.Linear(10, 1)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        # x = self.act(self.fc3(x))
        return x
start = time.time()


lr = 0.03
epochs = 5000

loss_list = []
train_size = 5000
batch_size = 50
loss_mat = []

n_neuron_list = [9]

x_train_data = torch.tensor(x_train_data,dtype = torch.float)
y_train_data = torch.tensor(y_train_data,dtype = torch.float)
x_test_data = torch.tensor(x_test_data,dtype = torch.float)
y_test_data = torch.tensor(y_test_data,dtype = torch.float)

dataset = TensorDataset(x_train_data, y_train_data)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for n_neuron_idx in range(len(n_neuron_list)):
    n_neuron = n_neuron_list[n_neuron_idx]
    model = MLP_Classifier(n_neuron)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for i in range(epochs):
        for batch_idx, samples in enumerate(dataloader):
            x_train_data, y_train_data = samples
            optimizer.zero_grad()
            pred = model(x_train_data)
                 
            loss = criterion(pred,y_train_data)
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.detach().numpy())
        
    trained_dict = model.state_dict()
    model = MLP_Classifier(n_neuron).to(device)
    tester(x_test_data, y_test_data, model, trained_dict)    
    loss_mat.append(loss_list)
end = time.time()
gap = end - start
print(gap)
fig, ax = plt.subplots(1,1, figsize = (15,15))
ax.grid()
ax.plot(loss_list)
ax.set_title("loss", fontsize = 50)

#fig, ax1 = plt.subplots(figsize = (10,10))
# ax2.plot(theta1_list, label = r"$\theta_{1}$")
# ax2.plot(theta0_list, label = r"$\theta_{0}$")
# fig.legend(fontsize = 'xx-large')
# ax2.set_title(r"$\theta_{1}, \theta_{0} Update$", fontsize = 50)
# ax2.grid()