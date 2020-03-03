import numpy as np
import matplotlib.pyplot as plt

import math
import sys
from hog_nb import Gradient2

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
        return self.histogram






data_x = np.load('./Sign-language-digits-dataset/X.npy')
data_y = np.load('./Sign-language-digits-dataset/Y.npy')
padding = 0
stride = 1
batch = (8,8)
input = data_x[0]



grad = Gradient(input = input, pad = padding, stride = stride)
histogram  = grad.auto()
print(type(histogram))








# fig, ax = plt.subplots(4,1,figsize = (30,30))
# ax[0].imshow(grad_mag,'gray')
# ax[1].imshow(grad_ang,'gray')
# ax[2].imshow(grad_mag_2,'gray')
# ax[3].imshow(grad_ang_2,'gray')




        









