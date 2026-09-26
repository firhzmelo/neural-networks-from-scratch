from classes.MyNeuralNetwork import MyNeuralNetworkClassifier
import numpy as np

X = np.array([1, 2, 3, 4, 5])
print(X.shape)

nn = MyNeuralNetworkClassifier([3, 4])

nn.fit(X, [1])


print(nn.forward(X))

