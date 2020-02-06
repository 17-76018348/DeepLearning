import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime

import torch
import torch.nn as nn
import torch.optim as optim


test_x = torch.tensor([1,5,3,4,2,9,6,7,8,0])
test_x = test_x.view(5,2)
print(test_x)
print(test_x.shape)
topv, topi = test_x.topk(1,dim = 1)
print(topv)
print(topi)
topi = topi.view(-1)
print(topi)