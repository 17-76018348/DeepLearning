import numpy as np
import matplotlib as plt

import math


class Gradient():
    def __init__(self,input,filter = "sobel"):
        if filter == "sobel":
            self.filter_x = np.array([[-1,0,1],
                             [-2,0,2],
                             [-1,0,1]]
            )
            self.filter_y = np.array([[1,2,1],
                             [0,0,0],
                             [-1,-2,-1]]
            )
            self.input = input
            self.fil_size = 3
            self.in_x = len(self.input)
            self.in_y = len(self.input)
            self.grad_x = np.zeros(
                                    (self.in_y-self.fil_size+1,
                                    self.in_x-self.fil_size+1)
            )
            self.grad_y = np.zeros_like(self.grad_x)
            set_grad()
            self.grad_mag = set_grad_mag()
            self.grad_ang = set_grad_ang()

    def set_grad(self):
        for i in range(self.in_y - self.fil_size + 1):
            for j in range(self.in_x - self.fil_size + 1):
                grad_x[i][j] = self.input[i:i+3][j:j+3] * self.filter_x
                grad_y[i][j] = self.input[i:i+3][j:j+3] * self.filter_y 
        
    def set_grad_mag(self):
        grad_mag = math.sqrt(pow(self.grad_x,2) + pow(self.grad_y,2))
        return grad_mag
        
    def set_grad_ang(self):
        grad_ang = math.atan2(self.grad_y,self.grad_x)
        return grad_ang
    def show_image(self,image):
        pass
# class hog():
#     def __init__(self):
#         pass
#     def cal_hist():
#         pass

# input image
input = []


        

        
        
        




