import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import basic_nodes as nodes
import affine_MSE
import pickle
from tqdm import trange
def get_data_batch(dataset, batch_idx, batch_size, n_batch):
    if batch_idx is n_batch -1:
        batch = dataset[batch_idx*batch_size:]
    else:
        batch = dataset[batch_idx*batch_size : (batch_idx+1)*batch_size]
    return batch
plt.style.use('seaborn')
np.random.seed(0)
#%%
n_sample = 200
# h_order = 3
# h_order = 5
h_order = 3
x_data1 = np.linspace(0.05, 1 - 0.05, n_sample).reshape(-1, 1)
y_data = np.sin(2*np.pi*x_data1) + 0.2*np.random.normal(0, 1, size = (n_sample,1))

x_data = np.zeros(shape = (n_sample, 1))
for order in range(1, h_order + 1):
    order_data = np.power(x_data1, order)
    x_data = np.hstack((x_data, order_data))

data = np.hstack((x_data, y_data))
 
#%%
batch_size = 32
n_batch = np.ceil(data.shape[0]/batch_size).astype(int)
feature_dim = x_data.shape[1]-1
Th = np.ones(shape = (feature_dim + 1,), dtype = np.float).reshape(-1, 1)

affine = affine_MSE.Affine_Function(feature_dim, Th)
cost = affine_MSE.MSE_Cost()

epochs, lr = 1, 0.1
th_accum = Th.reshape(-1, 1)
cost_list = []
pickle_epoch = 0
#%%

# with open('h11_50001pickle_3.p','rb') as file:
#     pickle_epoch = pickle.load(file)
#     th_accum = pickle.load(file)
#     cost_list = pickle.load(file)
#     affine = pickle.load(file)
#     print('loaded epoch is : '+ str(pickle_epoch))
#%%

for epoch in trange(epochs-pickle_epoch):
    np.random.shuffle(data)

    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx, batch_size, n_sample)
        X, Y = batch[:,:-1], batch[:,-1]
        Pred = affine.forward(X)
        J = cost.forward(Y, Pred)
        
        dPred = cost.backward()
        affine.backward(dPred, lr)
        
        th_accum = np.hstack((th_accum, affine._Th))
        cost_list.append(J)


    if epoch % 50000 == 1:
        plt.clf()

        plt.scatter(x_data1,y_data,color = 'r')
        tmp_Th = copy.deepcopy(affine._Th)
        tmp_Th = tmp_Th.reshape((h_order+1))
        tmp_y_data = []
        for i in range(200):
            tmp_y_data.append(np.sum(x_data[i] * tmp_Th))
        plt.plot(x_data1,tmp_y_data)
        # plt.show()
        plt.savefig('h21_'+str(epoch) + 'figure_4.png')
        with open('h21_'+str(epoch) + 'pickle_4.p','wb') as file:
            pickle.dump(epoch,file)
            pickle.dump(th_accum,file)
            pickle.dump(cost_list,file)
            pickle.dump(affine,file)
#%%
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
#%%
# plt.scatter(x_data1,y_data)


# tmp_Th = copy.deepcopy(affine._Th)
# tmp_Th = tmp_Th.reshape((6))
# tmp_y_data = []
# for i in range(200):
#     tmp_y_data.append(np.sum(x_data[i] * tmp_Th))
# plt.plot(x_data1,tmp_y_data)
# # plt.show()
# plt.savefig('1.png')