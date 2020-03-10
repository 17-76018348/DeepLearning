import numpy as np
import matplotlib.pyplot as plt

import math
from tqdm import trange
import random


import torch
import torch.nn as nn
import torch.optim as optim




if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def zero_padding(pad_size, img):
    input_y = len(img)
    input_x = len(img[0])
    output = np.zeros((input_y + 2 * pad_size, input_x + 2 * pad_size))
    for y in range(input_y + 2 * pad_size):
        for x in range(input_x + 2 * pad_size):
            if pad_size <=  y < (input_y + pad_size) and pad_size <=  x < (input_x + pad_size): 
                output[y][x] = img[y-pad_size][x-pad_size]
    return output

def set_histogram(mag, ang):
    hist = np.zeros(shape = 10, dtype = float)
    for cnt in range(9):
        idx = np.where(ang < (cnt + 1) * 20)
        tmp2 = (ang[idx] - cnt * 20)/((cnt + 1) * 20)
        tmp1 = 1 -  tmp2
        tmp2 *= mag[idx] 
        tmp1 *= mag[idx]
        hist[cnt] += np.sum(tmp1)
        hist[cnt+1] += np.sum(tmp2)
        ang[idx] = 300
    hist[0] += hist[9]
    return hist[:9]

def hist_normalize(histogram, bat_h,bat_w):
    
    hist_h, hist_w, ang_num = histogram.shape
    ## 8,8,9
    # histogram = list(histogram)
    # output = np.zeros(shape = (hist_h - bat_h + 1,hist_w - bat_w + 1))
    output = []
    ## 7, 7
    batch = np.zeros(shape = (bat_h,bat_w,ang_num))
    batch_sum = np.zeros(shape = ())
    
    ## 2, 2
    for h in range(hist_h - bat_h + 1):   # 7
        for w in range(hist_w - bat_w + 1):   # 7
            batch = histogram[h:h+bat_h,w:w+bat_w,:]
            batch = batch.reshape((bat_h * bat_w,ang_num))
            batch_sum = batch[0] + batch[1] + batch[2] + batch[3]
            batch_sum = batch_sum / np.linalg.norm(batch_sum, axis = -1, ord = 2)
            output.append(batch_sum)
            
    output = np.array(output)
    return output





def plot_hist(hist):
    ## shape = [49][9] 
    cell_num, angs  = hist.shape
    # hist[0] 먼저 plot

    cnt = 0
    # shape = [9]
    # plt.figure(figsize = (20,20))
    fig, ax = plt.subplots(nrows = 7, ncols = 7, sharex = True, sharey = True, figsize = (10,10))
    for h in range(7):
        for w in range(7):
            for idx, val in enumerate(hist[cnt]):
                x = np.linspace(-2, 2,50)

                if val > 0.2:
                    line = ax[h][w].plot(x, np.tan((idx * 20 + 90) * np.pi / 180) * x)
                    
                    plt.setp(line, color = 'r', linewidth = 2.0 * val )
                    ax[h][w].axis('off')
            plt.xlim(-2,2)
            plt.ylim(-2,2)

            cnt += 1
    plt.show()
    
        

    
    
    
    
    





class Gradient():
    def __init__(self,input,pad,stride = 1,batch = (8,8),filter = "sobel"):
        if filter == "sobel":
            self.filter_x = np.array([[-1,0,1],
                                      [-2,0,2],
                                      [-1,0,1]]
            )
            self.filter_y = np.array([[1,2,1],
                                      [0,0,0],
                                      [-1,-2,-1]]
            )
            self.fil_size = 3
        self.pad = pad
        self.batch = batch
        self.bat_y = batch[0]
        self.bat_x = batch[1]
        self.input = input
        self.stride = stride
        self.in_x = len(self.input[0])
        self.in_y = len(self.input)
        self.grad_x = np.zeros(shape = (int(math.floor((self.bat_y + 2 * self.pad - self.fil_size)/self.stride) + 1),
                                        int(math.floor((self.bat_x + 2 * self.pad - self.fil_size)/self.stride) + 1)
            ))
        self.grad_y = np.zeros_like(self.grad_x)
        self.histogram = []
        
        

    def set_grad(self,img):

        for idx_h,h in enumerate(list(range(0, self.bat_y - self.fil_size + 2 * self.pad + 1, self.stride))):
            for idx_w,w in enumerate(list(range(0, self.bat_x - self.fil_size + 2 * self.pad + 1, self.stride))):

                self.grad_x[idx_h][idx_w] = np.sum(img[h:h+3,w:w+3] * self.filter_x)
                self.grad_y[idx_h][idx_w] = np.sum(img[h:h+3,w:w+3] * self.filter_y) 
        return self.grad_x,self.grad_y
    def set_grad_mag(self):
        grad_mag = np.power((np.power(self.grad_x,2) + np.power(self.grad_y,2)),1/2)
        return grad_mag
        
    def set_grad_ang(self):
        grad_ang = np.abs(np.arctan2(self.grad_y,self.grad_x+0.00000001))/np.pi*180
        return grad_ang
    def auto(self):
        for y in range(int(self.in_y/self.bat_y)):
            for x in range(int(self.in_x/self.bat_x)):
                img = input[y * self.bat_y: (y+1) * self.bat_y,x * self.bat_x: (x+1) * self.bat_x]
                # img = zero_padding(2,img)
                self.set_grad(img)
                self.grad_mag = self.set_grad_mag()
                self.grad_ang = self.set_grad_ang()
                self.histogram.append(set_histogram(self.grad_mag,self.grad_ang))
        self.histogram = np.array(self.histogram)
        self.histogram = self.histogram.reshape((int(self.in_y/self.bat_y), int(self.in_x/self.bat_x),9))
        return self.histogram

class Hog_MLP(nn.Module):
    def __init__(self, p):
        super(Hog_MLP, self).__init__()
        self.model = nn.Sequential(
                # nn.Dropout(p = p),
                nn.Linear(49 * 9,256),
                nn.ReLU(),
                # nn.Dropout(p = p),
                nn.Linear(256,64),
                nn.ReLU(),

                nn.Linear(64,10),
                nn.LogSoftmax(dim = 1)
            )
        
        
    def forward(self, x):
        x = self.model(x)
        return x




data_x = np.load('./Sign-language-digits-dataset/X.npy')
data_y = np.load('./Sign-language-digits-dataset/Y.npy')
padding = 0
stride = 1
batch = (8,8)


shuffle_idx = np.arange(data_x.shape[0])
np.random.shuffle(shuffle_idx)

data_x = data_x[shuffle_idx]
data_y = data_y[shuffle_idx]



# data_x, data_y = np.array(list(zip(*xy_data)))
# data_x = data_x.reshape(-1,64,64)
# data_y = data_y.reshape(-1,10)

img_num, h, w  = data_x.shape
img_num, label = data_y.shape
# pi



#%%
hist_list = []

for idx, img in enumerate(data_x):
    input = img
    grad = Gradient(input = input, pad = padding, stride = stride)
    histogram  = grad.auto()
    hist_normalized = hist_normalize(histogram,2,2)
    hist_list.append(hist_normalized)
    if idx % 100 == 0:
        print(idx)
        
 #%%
hist_list = torch.tensor(hist_list, dtype = torch.float)


data_y =  np.argmax(data_y, axis = 1)


data_y = torch.tensor(data_y, dtype = torch.long).view(-1,1)




#%%

# plt.imshow(input)
# plot_hist(hist_normalized)

#%%

epochs = 30

lr = 0.001
cnt = 0
loss_list = []
val_acc_list = []

model = Hog_MLP(p = 0).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr = lr)

train_x = hist_list[ :1800,: ,: ] 
test_x = hist_list[1800: ,: ,:]
train_y = data_y[:1800,: ]
test_y = data_y[1800: ,: ]





for epoch in trange(epochs):
        model.train()
        loss_epoch = 0
        for step, hist in enumerate(train_x):
            hist = hist.view(-1,49 * 9).to(device)
            label = train_y[step].to(device)
            pred = model(hist)
            optimizer.zero_grad()
            loss = criterion(pred,label)
            loss_epoch += loss.item() * pred.shape[0]
            loss.backward()
            optimizer.step()
        loss_epoch /= len(train_x)
        loss_list.append(loss_epoch)
        
        
        model.eval()
        val_acc = 0
        
        for step, hist in enumerate(test_x):
            hist = hist.view(-1,49*9).to(device)
            label = test_y[step].to(device)
            
            pred = model(hist)
            topv, topi = pred.topk(1, dim = 1)
            n_correct = (topi.view(-1) == label).type(torch.int)
            val_acc += n_correct.sum().item()
        val_acc /= len(test_x)
        val_acc_list.append(val_acc)
        
#%%
    
fig, ax = plt.subplots(2, 1, figsize = (30, 15))
ax[0].plot(loss_list)
ax[1].plot(val_acc_list)


