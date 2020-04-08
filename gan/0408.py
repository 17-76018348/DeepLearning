#%%
gan = GAN(input_dim = (28,28,1)
            , discriminator_conv_filters = [64,64,128,128]
            , discriminator_conv_kernel_size = [5,5,5,5]
            , discriminator_conv_strides = [2,2,2,1]
    )

#%%
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


class Gan(nn.Module):
    def __init__(self):
        super(Gan, self).__init__()
        




