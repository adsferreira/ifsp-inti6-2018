from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, 2 * np.pi, 0.05)
D = np.sin(X) * np.cos(2 * X)
X = np.reshape(X, (X.shape[0], 1))

D2 = np.sin(X / 2) * np.cos(3 * X)


plt.plot(X, D2, '-')
plt.show()