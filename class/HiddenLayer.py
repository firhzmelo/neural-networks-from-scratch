import numpy as np

class HiddenLayer:
    
    def relu(x : np.number) -> np.number:
            return max(0, x)

    def sigmoid(x : np.number) -> np.number:
        return 1 / (1 + np.exp(-x))

    def tanh(x : np.number) -> np.number:
        return np.tanh(x)
    
    activation_functions = {
         "relu": relu,
         "tanh": tanh,
         "sigmoid": sigmoid
    }

    def __init__(self, size: int, nxt_size: int, activation_ = "relu", is_first = False):
        n = size
        m = nxt_size
        first: bool = is_first 
        weight = np.random.rand(n, m)
        activation:str = activation_

    def act_function(self, x : np.number) -> np.number:

        for act in self.activation_functions.keys:
            if(act == self.activation):
                return self.activation_functions[act](x)