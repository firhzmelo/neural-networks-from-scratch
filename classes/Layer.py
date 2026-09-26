import numpy as np

class Layer:
    
    def softmax(x: np.ndarray) -> np.ndarray:
        return x / np.sum(np.exp(x))
    
    def relu(x: np.ndarray) -> np.ndarray:
            return np.maximum(0, x)

    activation_functions = {
         "relu": relu,
         "softmax": softmax
    }

    def __init__(self, layer_size_: int, prev_size_: int, activation:str = "relu"):
        self.size = layer_size_ 
        self.prev_size = prev_size_
        self.bias = np.random.rand(self.size, 1)
        self.weight = np.random.rand(self.size, self.prev_size)
        self.activation = activation

    def act_function(self, x : np.number) -> np.number:
        for act in self.activation_functions.keys():
            if(act == self.activation):
                return self.activation_functions[act](x)

    def calc_output(self, X):
        z = (self.weight @ X) + self.bias
        return self.act_function(z)