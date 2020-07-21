import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import basic_nodes as nodes
import os
import sys
utils_path = os.path.dirname(os.path.abspath(__name__)) + '/../utils/'
if utils_path not in sys.path:    
    sys.path.append(utils_path)


from LR_dataset_generator import LR_dataset_generator
    
plt.style.use('seaborn')
np.random.seed(0)

def get_data_batch(dataset, batch_idx, batch_size, n_batch):
    if batch_idx is n_batch -1:
        batch = dataset[batch_idx*batch_size:]
    else:
        batch = dataset[batch_idx*batch_size : (batch_idx+1)*batch_size]
    return batch

#%%
np.random.seed(0)

##### Start Your Code(Dataset Setting) #####
coefficient_list = [3, 3, 3, 3]
distribution_params = {1:{'mean':0, 'std':1},
                       2:{'mean':0, 'std':1},
                       3:{'mean':0, 'std':1},
                       }
##### End Your Code(Dataset Setting) #####


##### Start Your Code(Dataset Generation) #####
data_gen = LR_dataset_generator(feature_dim = 3)
data_gen.set_coefficient(coefficient_list)
data_gen.set_distribution_params(distribution_params)
x_data, y_data = data_gen.make_dataset()
dataset = np.hstack((x_data,y_data))
##### End Your Code(Dataset Generation) #####

#%%
feature_dim = 3
node1 = [None] + [nodes.mul_node() for _ in range(feature_dim)]
node2 = [None] + [nodes.plus_node() for _ in range(feature_dim)]
node3 = nodes.minus_node()
node4 = nodes.square_node()


node5 = nodes.mean_node()
##### End Your Code(Loss Implementation) #####

#%%
th3, th2, th1, th0 = 0.1, 0.1, 0.1, 0.1
batch_size = 8
batch_idx = 0
n_batch = int(np.ceil(dataset.shape[0]/batch_size))

batch = get_data_batch(dataset, batch_idx,batch_size, n_batch)
print("batch.shape : ", batch.shape)
X, Y = batch[:,1:-1], batch[:,-1]
print("X.shape : ", X.shape)
print("Y.shape : ", Y.shape, '\n')
#%%
##### Start Your Code(Learning Preparation) #####

lr = 0.01
epochs = 20

batch_size = 12
##### End Your Code(Learning Preparation) #####
th_list = [0.1, 0.1, 0.1, 0.1]
th_accum = np.array(th_list).reshape(-1, 1)
# th3_list, th2_list, th1_list, th0_list = [], [], [], []
cost_list = []
#%%
n_batch = int(np.ceil(dataset.shape[0]/batch_size))
for epoch in range(epochs):
    np.random.shuffle(dataset)
    for batch_idx in range(n_batch):
        ##### Start Your Code(Batch Extraction) #####
        batch = get_data_batch(dataset, batch_idx, batch_size, n_batch)
        X, Y = batch[:,1:-1], batch[:,-1]
        # ##### Start Your Code(Batch Extraction) #####
        
        # ##### Start Your Code(Forward Propagation) #####
        # Z1 = node1.forward(th1, X)
        # Z2 = node2.forward(th0, Z1)
        # Z3 = node3.forward(Y, Z2)
        # Z4 = node4.forward(Z3)
        # J = node5.forward(Z4)
        # ##### End Your Code(Forward Propagation) #####
        
        
        # ##### Start Your Code(Backpropagation) #####
        # dZ4 = node5.backward(1)
        # dZ3 = node4.backward(dZ4)
        # _, dZ2 = node3.backward(dZ3)
        # dTh0, dZ1 = node2.backward(dZ2)
        # dTh1, _ = node1.backward(dZ1)
        # ##### End Your Code(Backpropagation) #####
        
        
        # th1_list.append(th1)
        # th0_list.append(th0)
        # cost_list.append(J)
        
        
        # ##### Start Your Code(Gradient Descent Method) #####
        # dth1 = np.sum(dTh1)
        # dth0 = np.sum(dTh0)
        
        # th1 = th1 - lr*dth1
        # th0 = th0 - lr*dth0
        # ##### Start Your Code(Gradient Descent Method) #####
        z1_list = [None] * (feature_dim + 1)
        z2_list, dz2_list, dz1_list, dth_list = z1_list.copy(), z1_list.copy(), z1_list.copy(), z1_list.copy()
        
        for node_idx in range(1, feature_dim + 1):
            z1_list[node_idx] = node1[node_idx].forward(th_list[node_idx], X[node_idx])
        
        z2_list[1] = node2[1].forward(th_list[0], z1_list[1])
        for node_idx in range(2, feature_dim + 1):
            z2_list[node_idx] = node2[node_idx].forward(z2_list[node_idx - 1], z1_list[node_idx])
        z3 = node3.forward(Y, z2_list[-1])
        z4 = node4.forward(z3)
        J = node5.forward(z4)
        #Forward Propagation end
        
        #Backward Propagation start
        dz4 = node5.backward(1)
        dz3 = node4.backward(dz4)
        _, dz2_last = node3.backward(dz3)
        dz2_list[-1] = dz2_last
        
        for node_idx in reversed(range(1, feature_dim + 1)):
            dz2, dz1 = node2[node_idx].backward(dz2_list[node_idx])
            dz2_list[node_idx - 1] = dz2
            dz1_list[node_idx] = dz1
        
        dth_list[0] = dz2_list[0]
        for node_idx in reversed(range(1, feature_dim + 1)):
            dth, _ = node1[node_idx].backward(dz1_list[node_idx])
            dth_list[node_idx] = dth
        #Backward Propagation end
        

        for th_idx in range(len(th_list)):
            th_list[th_idx] = th_list[th_idx] - lr*dth_list[th_idx]
        th_next = np.array(th_list).reshape(-1, 1)
        th_accum = np.hstack((th_accum, th_next))
        cost_list.append(J)
# fig, ax = plt.subplots(2, 1, figsize = (15,8))
# fig.subplots_adjust(hspace = 0.3)
# ax[0].plot(th1_list)
# ax[0].plot(th0_list)
# ax[1].plot(cost_list)
# ax[0].tick_params(axis = 'both', labelsize = 20)
# ax[1].tick_params(axis = 'both', labelsize = 20)
# ax[0].set_title(r'$\theta$', fontsize = 30)
# ax[1].set_title(r'$\mathcal{L}$', fontsize = 30)
fig, ax = plt.subplots(2, 1, figsize = (40, 20))

for i in range(feature_dim + 1):
    ax[0].plot(th_accum[i], label = r'$\theta_{%d}$'%i,
               linewidth = 5)
ax[1].plot(cost_list)
ax[0].legend(loc = 'lower right',
            fontsize = 30)
ax[0].tick_params(axis = 'both', labelsize = 30)
ax[1].tick_params(axis = 'both', labelsize = 30)

ax[0].set_title(r'$\vec{\theta}$', fontsize = 40)
ax[1].set_title('Cost', fontsize = 40)