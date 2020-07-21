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
coefficient_list = [5, 2, 4, 1]
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


#%%
th3, th2, th1, th0 = 0.1, 0.1, 0.1, 0.1
batch_size = 8
batch_idx = 0
n_batch = int(np.ceil(dataset.shape[0]/batch_size))

batch = get_data_batch(dataset, batch_idx,batch_size, n_batch)
print("batch.shape : ", batch.shape)
X, Y = batch[:,:-1], batch[:,-1]
print("X.shape : ", X.shape)
print("Y.shape : ", Y.shape, '\n')
#%%
##### Start Your Code(Learning Preparation) #####

lr = 0.01
epochs = 20

batch_size = 8
##### End Your Code(Learning Preparation) #####
Th_list = [0.1, 0.1, 0.1, 0.1]
th_accum = np.array(Th_list).reshape(-1, 1)
# th3_list, th2_list, th1_list, th0_list = [], [], [], []
cost_list = []
#%%
n_batch = int(np.ceil(dataset.shape[0]/batch_size))
for epoch in range(epochs):
    np.random.shuffle(dataset)
    for batch_idx in range(n_batch):
        ##### Start Your Code(Batch Extraction) #####
        batch = get_data_batch(dataset, batch_idx, batch_size, n_batch)
        X, Y = batch[:,:-1], batch[:,-1]

        Z1_list = [None] * (feature_dim + 1)
        Z2_list = Z1_list.copy()
        dZ1_list, dZ2_list = Z1_list.copy(), Z1_list.copy()
        dTh_list = dZ1_list.copy()
        for node_idx in range(1, feature_dim + 1):
            Z1_list[node_idx] = node1[node_idx].forward(Th_list[node_idx], X[:,node_idx])
        
        Z2_list[1] = node2[1].forward(Th_list[0], Z1_list[1])
        for node_idx in range(2, feature_dim + 1):
            Z2_list[node_idx] = node2[node_idx].forward(Z2_list[node_idx - 1], Z1_list[node_idx])
        Z3 = node3.forward(Y, Z2_list[-1])
        Z4 = node4.forward(Z3)
        J = node5.forward(Z4)
        #Forward Propagation end
        
        #Backward Propagation start
        dZ4 = node5.backward(1)
        dZ3 = node4.backward(dZ4)
        _, dZ2_last = node3.backward(dZ3)
        dZ2_list[-1] = dZ2_last
        
        for node_idx in reversed(range(1, feature_dim + 1)):
            dZ2, dZ1 = node2[node_idx].backward(dZ2_list[node_idx])
            dZ2_list[node_idx - 1] = dZ2
            dZ1_list[node_idx] = dZ1
        
        dTh_list[0] = dZ2_list[0]
        for node_idx in reversed(range(1, feature_dim + 1)):
            dTh, _ = node1[node_idx].backward(dZ1_list[node_idx])
            dTh_list[node_idx] = dTh
        #Backward Propagation end
        

        for th_idx in range(len(Th_list)):
            Th_list[th_idx] = Th_list[th_idx] - lr*np.sum(dTh_list[th_idx])
        th_next = np.array(Th_list).reshape(-1, 1)
        th_accum = np.hstack((th_accum, th_next))
        cost_list.append(J)

fig, ax = plt.subplots(2, 1, figsize = (20, 20))

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