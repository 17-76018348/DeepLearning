import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import os
import sys
utils_path = os.path.dirname(os.path.abspath(__name__)) + '/../utils/'
if utils_path not in sys.path:    
    sys.path.append(utils_path)

from LR_dataset_generator import LR_dataset_generator

plt.style.use('seaborn')
np.random.seed(0)
##### Start Your Code(Learning Preparation) #####
n_sample = 200
coefficient_list = [7, -3, 5]
data_gen = LR_dataset_generator(feature_dim = 2)
data_gen.set_n_sample(n_sample)
distribution_params = {
    1:{'mean':100, 'std':1},
    2: {'mean':10, 'std':1}
}
data_gen.set_distribution_params(distribution_params)
data_gen.set_coefficient(coefficient_list)
x_data, y_data = data_gen.make_dataset()
dataset = np.hstack((x_data, y_data))
##### End Your Code(Learning Preparation) #####
print(dataset.shape)
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import sys
utils_path = os.path.dirname(os.path.abspath(__name__)) + '/../utils/'
if utils_path not in sys.path:    
    sys.path.append(utils_path)

import basic_nodes as nodes
from LR_dataset_generator import LR_dataset_generator
    
plt.style.use('seaborn')
np.random.seed(0)

# Datset Setting
coefficient_list = [-1, 2, 100]
distribution_params = {1:{'mean':100, 'std':1}}

# Dataset Generation
data_gen = LR_dataset_generator(feature_dim = 2)
data_gen.set_coefficient(coefficient_list)
data_gen.set_distribution_params(distribution_params)
x_data, y_data = data_gen.make_dataset()
dataset = np.hstack((x_data, y_data))
##### End Your Code(Learning Preparation) #####
print(dataset.shape)
#%%


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import os
import sys
utils_path = os.path.dirname(os.path.abspath(__name__)) + '/../utils/'
if utils_path not in sys.path:    
    sys.path.append(utils_path)

from LR_dataset_generator import LR_dataset_generator

plt.style.use('seaborn')
np.random.seed(0)

##### Start Your Code(Learning Preparation) #####
n_sample = 200
coefficient_list = [7, -3, 5]
data_gen = LR_dataset_generator(feature_dim = 2)
data_gen.set_n_sample(n_sample)
distribution_params = {
    1: {'mean':0, 'std':1},
    2: {'mean':100, 'std':1}
}
data_gen.set_coefficient(coefficient_list)
x_data, y_data = data_gen.make_dataset()
dataset = np.hstack((x_data, y_data))
##### End Your Code(Learning Preparation) #####
print(dataset.shape)
##### Start Your Code(Learning Preparation) #####
th2, th1, th0 = 0.1, 0.1, 0.1
lr = 0.01
epochs = 3
##### End Your Code(Learning Preparation) #####

th2_list, th1_list, th0_list = [], [], []
loss_list = []

for epoch in range(epochs):
    for data_sample in dataset:

        
        ##### Start Your Code(Forward Propagation) #####
        pred = th2*x2 + th1*x1 + th0
        loss = np.power(y - pred, 2)
        ##### Start Your Code(Forward Propagation) #####
        th2_list.append(th2)
        th1_list.append(th1)
        th0_list.append(th0)
        loss_list.append(loss)
        
        ##### Start Your Code(Gradient Descent Method) #####
        th2 = th2 + 2*x2*lr*(y - pred)
        th1 = th1 + 2*x1*lr*(y - pred)
        th0 = th0 + 2*lr*(y - pred)
        ##### Start Your Code(Gradient Descent Method) #####
        
        
fig, ax = plt.subplots(2, 1, figsize = (20,10))
ax[0].plot(th2_list, label = r'$\theta_{2}$')
ax[0].plot(th1_list, label = r'$\theta_{1}$')
ax[0].plot(th0_list, label = r'$\theta_{0}$')
ax[0].legend(loc = 'upper left', fontsize = 30)
ax[1].plot(loss_list)
ax[0].set_title(r'$\theta$', fontsize = 30)
ax[1].set_title(r'$\mathcal{L}$', fontsize = 30)
for ax_idx in range(2):
    ax[ax_idx].tick_params(axis = 'both', labelsize = 20)










