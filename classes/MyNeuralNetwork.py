import numpy as np
from classes.Layer import Layer

class MyNeuralNetworkClassifier:

    def __init__(
            self, layers_size: list[int],
            activation: str = "relu",
            max_iter_: int = 1000,
            lr:float = 0.01
        ):
        self.n_classes = 0
        self.class_names = []
        prev_size = [0] + layers_size[:-1]
        self.layers = [Layer(s, prev) for s, prev in zip(layers_size, prev_size)]
        self.max_iter = max_iter_

    
    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.calc_output(output)
        return output

    def calc_loss(self, X, y):
        N = X.shape[0]
        output_proba = self.forward(X)
        L = -1 / N * np.dot(y, np.log(output_proba))
        return L

    def gradient(self, X, y):
        pass

    def fit(self, X, y):
        X = X.T
        self.class_names = np.unique(y)
        self.n_classes = len(self.class_names)
        s = self.layers[0].size
        self.layers[0] = Layer(s, X.shape[0])
        self.layers.append(Layer(self.n_classes, self.layers[-1].size, activation_="softmax"))
        for i in range(self.max_iter):
            pass
            self.gradient(X, y)


    