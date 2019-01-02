from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


X = np.arange(0, 2 * np.pi, 0.05)
D = np.sin(X) * np.cos(2 * X)
X = np.reshape(X, (X.shape[0], 1))

X_train, X_test, D_train, D_test = train_test_split(X, D, test_size=0.2)

# scale input data
scaler = StandardScaler() 
scaler.fit(X)
X = scaler.transform(X)  

# train MLP
mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='tanh', solver='lbfgs')
mlp.fit(X, D)

# mean square error for training data
mlp_training_answers = mlp.predict(X)
training_mse = mean_squared_error(D, mlp_training_answers)
print("Training mse: %.3f" %(training_mse))

# plot results
org_training_data_plot = plt.plot(X, D, '-')
mlp_training_data_plot = plt.plot(X, mlp_training_answers, '--')
plt.setp(org_training_data_plot, color = 'r', linewidth = 1.5)
plt.setp(mlp_training_data_plot, color = 'b', linewidth = 1.5)
plt.xlim(min(X), max(X))
plt.ylim(min(D) - 0.1, max(D) + 0.1)
plt.title("sen(x)cos(2x)")
plt.grid()
plt.show()