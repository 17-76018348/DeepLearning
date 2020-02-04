##### Check Overfitting for various n_neuron
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime

import torch
import torch.nn as nn
import torch.optim as optim

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
x_train_data = torch.tensor(x_train_data,dtype = torch.float)
y_train_data = torch.tensor(y_train_data,dtype = torch.float)
x_test_data = torch.tensor(x_test_data,dtype = torch.float)
y_test_data = torch.tensor(y_test_data,dtype = torch.float)
class MLP_Classifier(nn.Module):
    def __init__(self,fc1_input_size,n_neuron,fc2_output_size):
        super(MLP_Classifier,self).__init__()
        self.fc1_input_size, self.n_neuron,self.fc2_output_size = fc1_input_size, n_neuron, fc2_output_size
        self.fc1 = nn.Linear(self.fc1_input_size,self.n_neuron)
        self.fc2 = nn.Linear(self.n_neuron,self.fc2_output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
print("0")
fc1_input_size = 2
lr = 0.01
fc2_output_size = 1
epochs = 10000
loss_list = []
n_neuron_list = [10,20,30]
print("1")
n_neuron = 3
model = MLP_Classifier(fc1_input_size,n_neuron,fc2_output_size)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr = lr)
for i in range(epochs):
    pred= model(x_train_data)
    optimizer.zero_grad()
    loss = criterion(pred,y_train_data)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.detach().numpy())


print("3")
model = MLP_Classifier(fc1_input_size,n_neuron,fc2_output_size).to(device)
   
trained_dict = model.state_dict()
model = MLP_Classifier(fc1_input_size,n_neuron,fc2_output_size).to(device)
tester(x_train_data, y_train_data, model, trained_dict)
print("4")