import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
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

x_train_data, y_train_data, x_test_data, y_test_data = dataset_generator()



x_train_data = torch.tensor(x_train_data,dtype = torch.float).view(-1,2)
y_train_data = torch.tensor(y_train_data,dtype = torch.long).view(-1)
x_test_data = torch.tensor(x_test_data,dtype = torch.float).view(-1,2)
y_test_data = torch.tensor(y_test_data,dtype = torch.long).view(-1)

# fig = plt.figure(figsize = (15,15))
# ax= fig.gca(projection = '3d')

# ax.plot()







class MLP_Classifier(nn.Module):
    def __init__(self,input_size,n_neuron,output_size):
        super(MLP_Classifier,self).__init__()
        self.input_size, self.n_neuron,self.output_size = input_size, n_neuron,output_size
        self.fc1 = nn.Linear(self.input_size,self.n_neuron)
        self.fc2 = nn.Linear(self.n_neuron,self.output_size)
        self.logsoftmax = nn.LogSoftmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
        self.nll = nn.NLLLoss()
        self.tanh = nn.Tanh()
    def forward(self,x,nll = True):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.logsoftmax(x)
        return x


input_size = 2
lr = 0.01
output_size = 2
epochs = 5000
loss_list = []
n_neuron = 10




model = MLP_Classifier(input_size,n_neuron,output_size)

model.train()
criterion = nn.NLLLoss()
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

#%%

trained_dict = model.state_dict()

model = MLP_Classifier(input_size,n_neuron,output_size)
model.load_state_dict(trained_dict)
model.eval()
pred_test = model(x_test_data)


print(pred_test)
print(y_test_data)



print("\n\n\n")


topv, topi = pred_test.topk(1,dim = 1)
print(topv)
print(topi)
print(topi.sum())
topi = topi.view(-1)

n_correct = (topi == y_test_data).to(float).sum()
print(n_correct)
print(n_correct/y_test_data.shape[0])

