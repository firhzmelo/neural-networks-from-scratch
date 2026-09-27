import numpy as np

class Layer:
    
    def softmax(x: np.ndarray) -> np.ndarray:
        print(x)
        return np.exp(x) / np.sum(np.exp(x))
    
    def relu(x: np.ndarray) -> np.ndarray:
            return np.maximum(0, x)

    activation_functions = {
         "relu": relu,
         "softmax": softmax
    }

    def __init__(
            self,
            layer_size_: int, prev_size_: int,
            activation:str = "relu", lr:float = 0.01
        ):
        self.size = layer_size_ 
        self.prev_size = prev_size_
        self.bias = np.random.rand(self.size, 1)
        self.weight = np.random.rand(self.size, self.prev_size)
        self.activation = activation
        self.lr = lr
        self.output: np.ndarray = None
        self.prev_layer_out:Layer = None
        self.nxt_layer_weight = None
        self.gradient = None

    def act_function(self, x):
        for act in self.activation_functions.keys():
            if(act == self.activation):
                return self.activation_functions[act](x)

    # We will use X to reffer to the input of the layer
    def calc_output(self, X):
        z = (self.weight @ X) + self.bias
        self.output = self.act_function(z)
        return self.output

    def softmax_gradient(self, y_hat, y):
        return ((-1 / y_hat) * (y_hat[y, :] - 1)).reshape((y_hat.shape))

    def relu_gradient(self, in_gradient): 
        return (in_gradient.T @ self.nxt_layer_weight).T * self.output 

    def update_param_softmax(self):
        self.weight = self.weight - self.lr * (self.gradient @ self.prev_layer_out.T)
        self.bias = self.bias - self.lr * np.sum(self.gradient, axis = 1, keepdims = True)

    def update_param_relu(self):
        self.weight = self.weight - self.lr * (self.gradient @ self.prev_layer_out.T)
        self.bias = self.bias - self.lr * (self.gradient)

    def calc_gradient(self, in_gradient, y_hat = None, y = None):
        if(self.activation == "softmax"):
            self.gradient = self.softmax_gradient(y_hat, y)
            self.update_param_softmax()
        else:
            self.gradient = self.relu_gradient(in_gradient)
            self.update_param_relu()
