from classes.MyNeuralNetwork import MyNeuralNetworkClassifier
import numpy as np

X = np.array([[1, 2, 3, 4, 5], [1, 1, 1, 3, 4]])
X = X.T
print(X.shape)

nn = MyNeuralNetworkClassifier([3, 4])

nn.fit(X, np.array([1, 0]))
for layer in nn.layers:
    print(layer.size, layer.prev_size)


nn.forward(X)