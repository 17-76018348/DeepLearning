##### Check Overfitting for various n_neuron
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from utils import dataset_generator, tester, sphere_dataset

np.random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

x_train_data, y_train_data, x_test_data, y_test_data = sphere_dataset()

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



x_train_data = torch.tensor(x_train_data,dtype = torch.float)
y_train_data = torch.tensor(y_train_data,dtype = torch.float)
x_test_data = torch.tensor(x_test_data,dtype = torch.float)
y_test_data = torch.tensor(y_test_data,dtype = torch.float)




class MLP_Classifier(nn.Module):
    def __init__(self,input_size,n_neuron,output_size):
        super(MLP_Classifier,self).__init__()
        self.input_size, self.n_neuron,self.output_size = input_size, n_neuron,output_size
        self.fc1 = nn.Linear(self.input_size,self.n_neuron)
        self.fc2 = nn.Linear(self.n_neuron,self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self,x):
        x = self.tanh(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
print("0")
input_size = 3
lr = 0.07
output_size = 1
epochs = 1000
loss_list = []
n_neuron = 10
n_neuron_2 = 3
print("1")

model = MLP_Classifier(input_size,n_neuron,output_size)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr = lr)
for i in range(epochs):
    pred= model(x_train_data)
    optimizer.zero_grad()
    loss = criterion(pred,y_train_data)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.detach().numpy())

fig, ax = plt.subplots(figsize = (20,20))
ax.plot(loss_list)
print("3")
   
trained_dict = model.state_dict()
model = MLP_Classifier(input_size,n_neuron,output_size)
# tester(x_test_data, y_test_data, model, trained_dict)

pred_test = model(x_test_data)
print(pred_test)
print(y_test_data)
# test_x = torch.normal(0,1,size = (10,2))
pred_test = pred_test.view(-1)
# pred = model(test_x)
print(y_test_data.shape)
# label = torch.tensor([2,0,0,2,1,2,1,3,2,1])

# topv, topi = pred.topk(1,dim = 1)
# topi = topi.view(-1)
n_correct = (pred_test == y_test_data).to(float).sum()
print(n_correct)
print(n_correct/y_test_data.shape[0])