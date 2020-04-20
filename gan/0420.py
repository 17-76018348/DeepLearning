import os
import struct 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import trange

import torch 
import torch.nn as nn
import torch.optim as optim 

from torchvision import datasets, transforms
from torchvision.datasets import MNIST 
from torch.utils.data import Datasets, DataLoader

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


input_dim = (28,28,1)
discriminator_conv_filters = [64,64,128,128]
discriminator_conv_kernal_size = [5,5,5,5]
discriminator_conv_strides = [2,2,2,1]
discriminator_batch_norm_momentum = None
discriminator_activation = 'relu'
discriminator_dropout_rate = 0.4
discriminator_learning_rate = 0.0008
generator_initial_dense_layer_size = (7, 7, 64) 
generator_upsample = [2,2, 1, 1]
generator_conv_filters = [128,64,64,1]
generator_conv_kernel_size = [5,5,5,5] 
generator_conv_strides = [1,1, 1, 1] 
generator_batch_norm_momentum = 0.9
generator_activation = 'relu'
generator_dropout_rate = None
generator_learning_rate = 0.0004 
optimiser = 'rmsprop' 
z_dim = 100 


class Gan(nn.Module):
    def __init__(self):
        super(Gan, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels=64,
                        kernel_size=(5,5), stride=2,
                         padding = 2),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels = 64,out_channels=64,
                     kernel_size=(5,5), stride = 2,
                      paddng = 2),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels = 64,out_channels=128,
                    kernel_size = (5,5), stride=2,padding=2),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels = 128, out_channels=128,
                    kernel_size=(5,5), stride = 1,padding = 2),
            nn.ReLU(True),
            nn.Dropout(0.4)
        )
        self.fcl_discriminator = nn.Sequential(
                    nn.Linear(2048,1),
                    nn.Sigmoid()
        )
        self.fcl_generator = nn.Sequential(
                    nn.Linear(100,3136),
                    

            

        )
#%%


