# -*- coding: utf-8 -*-
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import csv

# X = np.arange(0, 2 * np.pi, 0.05)
# D = np.sin(X) * np.cos(2 * X)
# X = np.reshape(X, (X.shape[0], 1))
with open('dados2.csv', 'r') as myfile:
    rd = csv.reader(myfile)
    patterns_train = np.array(list(rd), dtype=float)
    
with open('dados3.csv', 'r') as myfile:
    rd = csv.reader(myfile)
    patterns_test = np.array(list(rd), dtype=float)

X_train = patterns_train[:,:-1]
D_train = patterns_train[:,-1]
X_test = patterns_test[:,:-1]
D_test = patterns_test[:,-1]
# split data into training and test patterns
#X_train, X_test, D_train, D_test = train_test_split(X, D, test_size=0.2)

# scale input data
scaler = StandardScaler() 
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

# train MLP
mlp = MLPRegressor(hidden_layer_sizes=(10), activation='tanh', solver='lbfgs')
mlp.fit(X_train, D_train)

# mean square error for training data
mlp_tr_answers = mlp.predict(X_train)
tr_mse = mean_squared_error(D_train, mlp_tr_answers)
print("Training mse: %.3f" %(tr_mse))
# mean square error for test data
mlp_te_answers = mlp.predict(X_test)
te_mse = mean_squared_error(D_test, mlp_te_answers)
print("Test mse: %.3f" %(te_mse))

fig = plt.figure()
ax = Axes3D(fig)
#X, Y = np.meshgrid(X_train[:,0], X_train[:,1])
X, Y = np.meshgrid(X_train[:,0], X_train[:,0])
#zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
#D_train = D_train.reshape(X.shape[0], 1)

#ax.plot_surface(X, Y, D_train)

# Plot the surface.
#surf = ax.scatter(X_train[:,0], X_train[:,1], D_train)
surf = ax.plot_trisurf(X_train[:,0], X_train[:,1], D_train, cmap=cm.coolwarm, linewidth=0)
plt.title("Figura 1: unknown function.")
# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# plot results
# original curve
# obtain ids of the sorted array
# sorted_ids = np.argsort(X_test, axis = 0)
# X_test = X_test[sorted_ids].reshape(-1, X_test.shape[1])
# D_test = D_test[sorted_ids].reshape(-1)
# mlp_te_answers = mlp_te_answers[sorted_ids].reshape(-1)
# orig_training_data_plot = plt.plot(X_test, D_test, 'ro')
# mlp_training_data_plot = plt.plot(X_test, mlp_te_answers, 'b+')
#plt.setp(line_tar, color = 'r', linewidth = 1.5)
#plt.setp(line_mlp, color = 'b', linewidth = 1.5)
# plt.xlim(min(X), max(X))
# plt.ylim(min(D) - 0.1, max(D) + 0.1)
# plt.title("sen(x)cos(2x)")
# plt.grid()
# plt.show()