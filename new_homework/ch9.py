import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import basic_nodes as nodes

def dataset_generator(x_dict):
    x_data = np.random.normal(x_dict['mean'], x_dict['std'],x_dict['n_sample'])
    x_data_noise = x_data + x_dict['noise_factor'] * np.random.normal(0,1,x_dict['n_sample'])
    
    if x_dict['direction'] > 0:
        y_data = (x_data_noise > x_dict['cutoff']).astype(np.int)
    else:
        y_data = (x_data_noise < x_dict['cutoff']).astype(np.int)
    
    data = np.zeros(shape = (x_dict['n_sample'],1))
    data = np.hstack((data,x_data.reshape(-1,1),y_data.reshape(-1,1)))
    return data
x_dict = {'mean':1, 'std':1, 'n_sample':300, 'noise_factor':0.3, 'cutoff':1, 'direction':1}
data = dataset_generator(x_dict)


node1 = nodes.mul_node()
node2 = nodes.plus_node()

Th = np.array([5., 5.]).reshape(-1,1)
th_accum = Th

loss_list = []
iter_idx, check_freq = 0, 5
epochs, lr = 100, 0.1

for epoch in range(epochs):
    np.random.shuffle(data)
    
    for data_idx in range(data.shape[0]):
        x,y = data[data_idx,1], data[data_idx,-1]
        
        z1 = node1.forward(Th[1],x)
        z2 = node2.forward(Th[0],z1)
        pred = 1/(1 + np.exp(-1*z2))
        
        loss = -1*(y*np.log(pred) + (1-y)*np.log(1 - pred))
        
        dpred = (pred - y) / (pred * (1 - pred))
        dz2 = dpred * (pred*(1-pred))
        dth0, dz1 = node2.backward(dz2)
        dth1, dx = node1.backward(dz1)
        
        Th[1] = Th[1] - lr*dth1
        Th[0] = Th[0] - lr*dth0
        
        if iter_idx % check_freq == 0:
            th_accum = np.hstack((th_accum, Th))
            loss_list.append(loss)
        iter_idx += 1
fig, ax = plt.subplots(2,1, figsize = (30,10))
fig.subplots_adjust(hspace = 0.3)
ax[0].set_title(r'$\vec{\theta}$' + 'Update')

ax[0].plot(th_accum[1,:], label = r'$\theta_{1}$')
ax[0].plot(th_accum[0,:], label = r'$\theta_{0}$')

ax[0].legend()
iter_ticks = np.linspace(0, th_accum.shape[1],10).astype(np.int)
ax[0].set_xticks(iter_ticks)

ax[1].set_title('Loss Decrease')
ax[1].plot(loss_list)
ax[1].set_xticks(iter_ticks)

n_pred = 1000
fig,ax = plt.subplots(figsize = (30,10))
ax.set_title('Predictor Update')
ax.scatter(data[:,1],data[:,-1])

ax_idx_arr = np.linspace(0, len(loss_list)-1, n_pred).astype(np.int)
cmap = cm.get_cmap('rainbow',lut = len(ax_idx_arr))

x_pred = np.linspace(np.min(data[:,1]),np.max(data[:,1]),1000)

for ax_cnt,ax_idx in enumerate(ax_idx_arr):
    z = th_accum[1, ax_idx]*x_pred + th_accum[0, ax_idx]
    a = 1/(1 + np.exp(-1*z))
    ax.plot(x_pred,a,color = cmap(ax_cnt),alpha = 0.2)
y_ticks = np.round(np.linspace(0,1,7),2)
ax.set_yticks(y_ticks)

