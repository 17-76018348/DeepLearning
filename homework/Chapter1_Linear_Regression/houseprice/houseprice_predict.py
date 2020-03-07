# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:22:29 2020

@author: didrm
"""

import numpy as np
import matplotlib.pyplot as plt
import Node as nd
file = open("Real estate.csv")

line_idx = 0
house_price = []
test_x = dict()
test_y = []
test_result = []
x_label = []
x_data = dict()
corr_xy = []
corr_xx = []
for line in file:
    line_split = line.strip().split(',')
    if line_idx == 0:
        for i in range(1,7):
            x_label.append(line_split[i])
            x_data[line_split[i]] = []
            test_x[line_split[i]] = []
    elif line_idx >= 1 and line_idx<370:
        for i in range(1,7):
            x_data[x_label[i-1]].append(float(line_split[i]))
        house_price.append(float(line_split[-1]))
    elif line_idx>=370:
        for i in range(1,7):
            test_x[x_label[i-1]].append(float(line_split[i]))
        test_y.append(float(line_split[-1]))
    line_idx += 1
    
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,10))
#ax1.grid()
#ax1.plot(x)
#ax1.set_title("loss", fontsize = 50)
    
#plt.plot(x_data[x_label[0]],house_price,'ro')
# plt.plot(x_data[x_label[5]],house_price,'ro')
# plt.hist(x_data[x_label[2]],rwidth = 0.8)
for i in range(len(x_label)):
   test_x[x_label[i]] = np.array(test_x[x_label[i]])
   test_x[x_label[i]] -= np.mean(x_data[x_label[i]])
   test_x[x_label[i]] /= np.std(x_data[x_label[i]])
   
   x_data[x_label[i]] = np.array(x_data[x_label[i]])
   x_data[x_label[i]] -= np.mean(x_data[x_label[i]])
   x_data[x_label[i]] /= np.std(x_data[x_label[i]])
   
house_price = np.array(house_price)    





print(np.corrcoef(x_data[x_label[3]],house_price))
for i in range(len(x_label)):
   corr_xy.append((np.corrcoef(x_data[x_label[i]],house_price))[0][1])
for idx1 in range(0,6):
   for idx2 in range(0,6):
       corr_xx.append(np.corrcoef(x_data[x_label[idx1]],x_data[x_label[idx2]])[0][1])
corr_xx = np.array(corr_xx).reshape(6,6,order = 'C')






theta4, theta3, theta2, theta1, theta0 = 0, 0, 0, 0, 0
lr = 0.0003
epochs = 12500

z4_node = nd.mul_node()
z3_node = nd.mul_node()
z2_node = nd.mul_node()
z1_node = nd.mul_node()
z5_node = nd.plus5_node()
z6_node = nd.minus_node()
loss_node = nd.square_node()
c_node = nd.cost_node()

loss_list = []

theta4_list,theta3_list,theta2_list,theta1_list, theta0_list = [], [], [], [], []

for i in range(epochs):
   # forward
   z4 = z4_node.forward(theta4,x_data[x_label[2]])
   z3 = z3_node.forward(theta3,x_data[x_label[3]])
   z2 = z2_node.forward(theta2,x_data[x_label[4]])
   z1 = z1_node.forward(theta1,x_data[x_label[5]])
   z5 = z5_node.forward(z1,z2,z3,z4,theta0)
   z6 = z6_node.forward(house_price,z5)
   loss = loss_node.forward(z6)
   cost = c_node.forward(loss)
   loss_list.append(cost)
   
   # backward
   dcost = c_node.backward()
   dloss = loss_node.backward(dcost)
   dy, dz6 = z6_node.backward(dloss)
   dz1,dz2,dz3,dz4,dtheta0 = z5_node.backward(dz6)
   dtheta4, dx4 = z4_node.backward(dz4)
   dtheta3, dx3 = z3_node.backward(dz3)
   dtheta2, dx2 = z2_node.backward(dz2)
   dtheta1, dx1 = z1_node.backward(dz1)
   
   theta4 -= lr*np.sum(dtheta4)
   theta3 -= lr*np.sum(dtheta3)
   theta2 -= lr*np.sum(dtheta2)
   theta1 -= lr*np.sum(dtheta1)
   theta0 -= lr*np.sum(dtheta0)
   
   theta4_list.append(theta4)
   theta3_list.append(theta3)
   theta2_list.append(theta2)
   theta1_list.append(theta1)
   theta0_list.append(theta0)
   
   
predict = theta4 * test_x[x_label[2]] + theta3 * test_x[x_label[3]]\
         +theta2 * test_x[x_label[4]] + theta1 * test_x[x_label[5]] + theta0
test_result =  test_y - predict
print("평균:    ",np.mean(test_result))




fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,10))
ax1.grid()
ax1.plot(loss_list)
ax1.set_title("loss", fontsize = 50)

#fig, ax1 = plt.subplots(figsize = (10,10))
ax2.plot(theta4_list, label = r"$\theta_{4}$")
ax2.plot(theta3_list, label = r"$\theta_{3}$")
ax2.plot(theta2_list, label = r"$\theta_{2}$")
ax2.plot(theta1_list, label = r"$\theta_{1}$")
ax2.plot(theta0_list, label = r"$\theta_{0}$")
fig.legend(fontsize = 'xx-large')
ax2.set_title(r"$\theta_{1}, \theta_{0} Update$", fontsize = 50)
ax2.grid()

plt.plot(test_result)
plt.grid()
plt.show()