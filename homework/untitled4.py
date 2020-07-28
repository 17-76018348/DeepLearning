import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#%%

##### Start Your Code(Data Sample Preparation) #####
x2 = 1
x1 = 1
y = 4
##### Start Your Code(Data Sample Preparation) #####
th2_range = np.linspace(-3, 5, 100)
th1_range = np.linspace(-3, 3, 100)
th0_range = np.linspace(-3, 3, 100)

Th2, Th1 = np.meshgrid(th2_range, th1_range)

##### Start Your Code(Loss Function) #####
loss = np.power(y - (Th2*x2 + Th1*x1 + 5), 2)
##### End Your Code(Loss Function) #####


fig, ax = plt.subplots(figsize = (7,7))
levels = np.geomspace(np.min(loss) + 0.01, np.max(loss), 30)
cmap = cm.get_cmap('Reds_r', lut = len(levels))
ax.contour(Th2, Th1, loss, levels = levels, cmap = cmap)