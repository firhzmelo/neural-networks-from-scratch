from classes.MyNeuralNetwork import MyNeuralNetworkClassifier
import numpy as np
X = np.random.rand(4, 6) 
y = np.array([1, 1, 2, 0])


nn = MyNeuralNetworkClassifier([8, 10], classes = 5, max_iter = 10)

nn.fit(X, y)

