import numpy as np
from classes.Layer import Layer

class MyNeuralNetworkClassifier:

    def __init__(
            self, layers_size: list[int],
            classes: int,
            class_names = [],
            max_iter: int = 1000,
            lr:float = 0.01,
            batch_size:int = 128 
        ):
        self.n_classes = classes
        self.class_names = class_names
        prev_size = [0] + layers_size[:-1]
        self.layers = [Layer(s, prev) for s, prev in zip(layers_size, prev_size)]
        self.max_iter = max_iter
        self.batch_size = batch_size

    
    def forward(self, X):
        output = X
        prev_layer = None
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
                layer.calc_gradient(1, y_hat)
            else:
                layer.calc_gradient(nxt_layer.gradient)
            nxt_layer = layer

    def calc_loss(self, y_hat, y):
        batch_size = y_hat.shape[-1]
        total_loss = -np.sum(np.log(y_hat[y, np.arange(batch_size)])) / batch_size
        return total_loss

    def fit(self, X, y):
        X = X.T
        #print("--", X.shape)
        s = self.layers[0].size
        self.layers[0] = Layer(s, X.shape[0])
        self.layers.append(Layer(self.n_classes, self.layers[-1].size, activation="softmax"))

        #for i, layer in enumerate(self.layers):
        #    print(i, ":", layer.size, layer.prev_size)

        for i in range(self.max_iter):
            y_hat = self.forward(X)
            total_loss = self.calc_loss(y_hat, y) 
            #print(total_loss)
            self.backward(y, y_hat)


    