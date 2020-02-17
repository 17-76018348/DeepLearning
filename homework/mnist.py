# import matplotlib.pyplot as plt
# import numpy as np
# train_img = open('./train-images.idx3-ubyte','rb')
# train_lab = open('./train-labels.idx1-ubyte','rb')
# test_img = open('./t10k-images.idx3-ubyte','rb')
# test_lab = open('./t10k-labels.idx1-ubyte','rb')
# img = train_img.read()
# lab = train_lab.read()
# t_img = test_img.read()
# t_lab = test_lab.read()
# print(len(img))
# print(len(lab))
# print(len(t_img))
# print(len(t_lab))
# print(lab[60007])



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
from torch.utils.data import Dataset, DataLoader



def read(dataset = "training", path = "."):

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    mnist_data = []
    for i in range(len(lbl)):
        mnist_data.append(get_img(i))
    return mnist_data


def show(image):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
    
    
mnist = read("training")
mnist = np.array(mnist)

# show(mnist[1][1])
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
    
class mnist(nn.Module):
    def __init__(self,input_data,n1_neuron,n2_neuron):
        super(mnist,self).__init__()
        self.input_data, self.n1_neuron, self.n2_neuron = input_data, n1_neuron, n2_neuron
        self.fc1 = nn.Linear(input_data,n1_neuron)
        self.fc2 = nn.Linear(n1_neuron,n2_neuron)
        self.
    

