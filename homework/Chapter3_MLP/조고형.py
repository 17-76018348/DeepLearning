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
    def __init__(self,fc1_input_size,fc1_output_size,fc2_output_size,n_neruon):
        super(MLP_Classifier,self).__init__()
        self.fc1_input_size, self.fc1_output_size,self.fc2_output_size = fc1_input_size, fc1_output_size, fc2_output_size
        self.fc1 = nn.Linear(self.fc1_input_size,self.n_neruon)
        self.n_neruon = n_neruon
        self.fc2 = nn.Linear(self.n_neruon,self.fc2_output_size)
    def forward(self,x):
        x = self.sigmoid(self.fc1.x)
        x = self.sigmoid(self.fc2.x)
        return x
n_neruon list = [10,20,30]

for n_idx in rnage(len(n_list)):
    n_neruon = n_neruon[n_nieron_idx]
    model = MLP(n_nerun)
    criterion - nn.BCELoss()
    oprimazier = optim.sgd
fc1_input_size = 2
fc1_output_size = 100
fc2_output_size = 1
model = MLP_Classifier(fc1_input_size,fc1_output_size,fc2_output_size).to(device)
   
trained_dict = model.state_dict()
model = MLP_Classifier(fc1_input_size,fc1_output_size,fc2_output_size).to(device)
tester(x_train_data, y_train_data, model, trained_dict)