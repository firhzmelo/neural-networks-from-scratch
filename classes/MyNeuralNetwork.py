import numpy as np
from classes.Layer import Layer

class MyNeuralNetworkClassifier:

    def __init__(
            self, layers_size: list[int],
            max_iter: int = 1000,
            lr:float = 0.01
        ):
        self.n_classes = 0
        self.class_names = []
        prev_size = [0] + layers_size[:-1]
        self.layers = [Layer(s, prev) for s, prev in zip(layers_size, prev_size)]
        self.max_iter = max_iter

    
    def forward(self, X):
        output = X
        prev_layer = None
        for layer in self.layers:
            layer.prev_layer = prev_layer
            output = layer.calc_output(output)
            prev_layer = layer
            print(output)
        return output

    def calc_loss(self, X, y):
        N = X.shape[0]
        output_proba = self.forward(X)
        L = -1 / N * np.dot(y, np.log(output_proba))
        return L

    def backward(self, X, y, predictions):
        nxt_layer:Layer = None
        for layer in reversed(self.layers):
            if(nxt_layer == None):
                layer.calc_gradient(1, predictions)
            else:
                layer.calc_gradient(nxt_layer.gradient)
            nxt_layer = layer

    def fit(self, X, y):
        X = X.T
        self.class_names = np.unique(y)
        self.n_classes = len(self.class_names)
        s = self.layers[0].size
        self.layers[0] = Layer(s, X.shape[-1])
        self.layers.append(Layer(self.n_classes, self.layers[-1].size, activation="softmax"))
        for i in range(self.max_iter):
            predictions = self.forward(X)
            self.backward(X, y)


    