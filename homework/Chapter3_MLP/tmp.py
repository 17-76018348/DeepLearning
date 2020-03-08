import numpy as np

test_x1 = np.linspace(-2, 2, 500)
test_x2 = np.linspace(-2, 2, 600)
X1, X2 = np.meshgrid(test_x1, test_x2)
print(X1.shape)
print(X2.shape)

test_X = np.dstack((X1, X2)).reshape(-1,2)
print(test_X.shape)
