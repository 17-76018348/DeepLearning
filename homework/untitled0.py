import matplotlib.pyplot as plt
import numpy as np

x_data = np.linspace(-10,10,num = 1000)
y_data = x_data ** 2

plt.plot(x_data,y_data)
plt.axis('off')
plt.show()