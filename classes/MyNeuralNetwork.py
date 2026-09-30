import numpy as np
from classes.Layer import Layer

class MyNeuralNetworkClassifier:

    def __init__(
            self, layers_size: list[int],
            input_size:int,
            classes: int,
            class_names = [],
            max_iter: int = 50,
            lr:float = 0.01,
            batch_size:int = 128 
        ):
        self.n_classes = classes
        self.class_names = class_names
        prev_size = [input_size] + layers_size[:-1]
        self.layers = [Layer(s, prev, lr) for s, prev in zip(layers_size, prev_size)]
        self.layers.append(Layer(self.n_classes, self.layers[-1].size, lr, activation="softmax"))
        self.max_iter = max_iter
        self.lr = lr
        self.batch_size = batch_size

    
    def forward(self, X):
        output = X
        for layer in self.layers:
            layer.prev_layer_out = output
            output = layer.calc_output(output)
        return output

    def backward(self, y, y_hat):
        nxt_layer:Layer = None
        for layer in reversed(self.layers):
            if nxt_layer != None:
                layer.nxt_layer_weight = nxt_layer.weight
            if(nxt_layer == None):
                layer.calc_gradient(1, y_hat, y)
            else:
                layer.calc_gradient(nxt_layer.gradient)
            nxt_layer = layer

    def calc_loss(self, y_hat, y):
        batch_size = y_hat.shape[-1]
        total_loss = -np.sum(np.log(y_hat[y, np.arange(batch_size)])) / batch_size
        return total_loss

    def fit(self, X, y, return_loss = False):
        X = np.array(X)
        y = np.array(y)
        X = X.T

        total_loss = None
        for i in range(self.max_iter):
            y_hat = self.forward(X)
            total_loss = self.calc_loss(y_hat, y) 
            self.backward(y, y_hat)
        if(return_loss):
            return total_loss


    def predict(self, X):
        X = np.array(X)
        X = X.T
        y_hat = self.forward(X)
        return y_hat.argmax(axis = 0).astype(np.int64)