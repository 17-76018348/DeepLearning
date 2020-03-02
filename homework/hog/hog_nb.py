import numpy as np

import math
class Gradient2():
    def __init__(self,input,pad,stride = 1,filter = "sobel"):
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
        self.input = input
        self.stride = stride
        self.in_x = len(self.input[0])
        self.in_y = len(self.input)

        self.grad_x = np.zeros(
                                (int(math.floor(self.in_y - self.fil_size)/self.stride + 1),
                                int(math.floor(self.in_x - self.fil_size)/self.stride + 1))
        )
        self.grad_y = np.zeros_like(self.grad_x)

    def set_grad(self):
        for h in range(0, self.in_y - self.fil_size + 1, self.stride):
            for w in range(0, self.in_x - self.fil_size + 1, self.stride):

                self.grad_x[h][w] = np.sum(self.input[h:h+3,w:w+3] * self.filter_x)
                self.grad_y[h][w] = np.sum(self.input[h:h+3,w:w+3] * self.filter_y) 
        
    def set_grad_mag(self):
        grad_mag = np.power((np.power(self.grad_x,2) + np.power(self.grad_y,2)),1/2)
        return grad_mag
        
    def set_grad_ang(self):
        grad_ang = np.abs(np.arctan2(self.grad_y,self.grad_x+0.00000001))/np.pi*180
        return grad_ang
    def auto(self):
        self.set_grad()
        self.grad_mag = self.set_grad_mag()
        self.grad_ang = self.set_grad_ang()
        return self.grad_mag, self.grad_ang