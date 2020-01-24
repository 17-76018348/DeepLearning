import numpy as np
import matplotlib.pyplot as plt
file = open("Real estate.csv")

line_idx = 0
house_price = []
x_label = []
x_data = dict()


for line in file:
    line_split = line.strip().split(',')
    if line_idx == 0:
        for i in range(1,7):
            x_label.append(line_split[i])
            x_data[line_split[i]] = []
    elif line_idx >= 1:
        for i in range(1,7):
            x_data[x_label[i-1]].append(float(line_split[i]))
        house_price.append(float(line_split[-1]))
    line_idx += 1
for i in range(len(x_label)):
    x_data[x_label[i]] = np.array(x_data[x_label[i]])
fig ,ax = plt.subplots(5,1,figsize = (25,25))

for fig_idx in range(5):
    ax[fig_idx].plot(x_data[x_label[fig_idx]],house_price,'bo')        