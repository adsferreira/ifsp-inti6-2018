from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, 2 * np.pi, 0.05)
D = np.sin(X) * np.cos(2 * X)
X = np.reshape(X, (X.shape[0], 1))

# scale input data
scaler = StandardScaler() 
scaler.fit(X)
X = scaler.transform(X) 

mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='tanh', solver='lbfgs')
mlp.fit(X, D)
# salva mlp modelada
joblib.dump(mlp, 'mlp_sklearn.pkl')
mlp_outputs = mlp.predict(X)
training_mse = mean_squared_error(D, mlp_outputs)
print("Training mse: %.3f" %(training_mse))

org_training_data_plot = plt.plot(X, D, 'ro')
mlp_training_data_plot = plt.plot(X, mlp_outputs, 'b+')
plt.xlim(min(X), max(X))
plt.ylim(min(D) - 0.1, max(D) + 0.1)
plt.grid()
plt.show()
