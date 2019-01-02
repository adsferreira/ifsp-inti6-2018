# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# create dataset
X = np.arange(0, 2 * np.pi, 0.05)
D = np.sin(X) * np.cos(2 * X)
X = np.reshape(X, (X.shape[0], 1))

# create training and test sets
# 15% of samples for testing
X_train, X_test, D_train, D_test = train_test_split(X, D, test_size=0.15)

# scale input data
scaler = StandardScaler() 
scaler.fit(X)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

# current number of hidden units
h_neurons = 1
max_nr_h_neurons = 50

nr_neurons_vec = []
mean_tr_mse_vec = []
mean_vl_mse_vec = []

for i in range(0, max_nr_h_neurons, 5):  
    sum_tr_mse = 0
    sum_vl_mse = 0  
    # k-fold cross-validation with k=5
    kf = KFold(n_splits=5,shuffle=True)
    for train, validation in kf.split(X_train):
        mlp = MLPRegressor(hidden_layer_sizes=(i + 1), activation='tanh', solver='lbfgs')
        # mlp training with training data
        mlp.fit(X_train[train], D_train[train])
        # mlp's prediction and mean square error for training data
        mlp_tr_answers = mlp.predict(X_train[train])
        tr_mse = mean_squared_error(D_train[train], mlp_tr_answers)
        # mlp's prediction mean square error for validation data
        mlp_vl_answers = mlp.predict(X_train[validation])
        vl_mse = mean_squared_error(D_train[validation], mlp_vl_answers)
        sum_tr_mse += tr_mse
        sum_vl_mse += vl_mse
    
    print("With %d neurons:" %(i + 1))
    print("mean training mse: %f - mean validation mse: %f" %(sum_tr_mse / 5, sum_vl_mse / 5))
    nr_neurons_vec.append(i + 1)
    mean_tr_mse_vec.append(sum_tr_mse / 5)
    mean_vl_mse_vec.append(sum_vl_mse / 5)

# plot error curves        
plt.plot(nr_neurons_vec, mean_tr_mse_vec, 'r-')
plt.plot(nr_neurons_vec, mean_vl_mse_vec, 'b-')
plt.show()