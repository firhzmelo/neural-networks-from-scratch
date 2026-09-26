import numpy as np

class Layer:
    
    def relu(x):
            return np.maximum(0, x)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def tanh(x):
        return np.tanh(x)
    
    def softmax(x):
        total = np.sum(np.exp(x))
        return np.exp(x) / total
    
    activation_functions = {
         "relu": relu,
         "tanh": tanh,
         "sigmoid": sigmoid,
         "softmax": softmax
    }


    def __init__(self, layer_size_: int, prev_size_: int, activation_: str = "relu"):
        self.size = layer_size_ 
        self.prev_size = prev_size_
        self.bias = np.random.rand(self.size)
        self.weight = np.random.rand(self.size, self.prev_size)
        self.activation:str = activation_

    def act_function(self, x : np.number) -> np.number:
        for act in self.activation_functions.keys():
            if(act == self.activation):
                return self.activation_functions[act](x)

    def calc_output(self, inputs):
        z = self.weight @ inputs + self.bias
        return self.act_function(z)