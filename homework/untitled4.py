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
coefficient_list = [3, 3]
distribution_params = {1:{'mean':0, 'std':1}}
##### End Your Code(Dataset Setting) #####


##### Start Your Code(Dataset Generation) #####
data_gen = LR_dataset_generator(feature_dim = 1)
data_gen.set_coefficient(coefficient_list)
data_gen.set_distribution_params(distribution_params)
x_data, y_data = data_gen.make_dataset()
##### End Your Code(Dataset Generation) #####