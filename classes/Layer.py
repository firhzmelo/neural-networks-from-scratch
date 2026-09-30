import numpy as np

class Layer:
    
    def softmax(x: np.ndarray) -> np.ndarray:
        maxi = np.max(x, axis = 0, keepdims = True)
        x_exp = np.exp(x - maxi)
        total = np.sum(x_exp, axis = 0)
        #print(x_exp/total)
        return x_exp / total
    
    def relu(x: np.ndarray) -> np.ndarray:
            return np.maximum(0, x)

    activation_functions = {
         "relu": relu,
         "softmax": softmax
    }

    def __init__(
            self,
            layer_size_: int, prev_size_: int,
            activation:str = "relu", lr:float = 0.001
        ):
        self.size = layer_size_ 
        self.prev_size = prev_size_
        self.bias = np.zeros((self.size, 1))
        self.weight = np.random.randn(self.size, self.prev_size) / np.sqrt(self.prev_size / 2)
        self.activation = activation
        self.lr = lr
        self.output: np.ndarray = None
        self.prev_layer_out:Layer = None
        self.nxt_layer_weight = None
        self.gradient = None

    def act_function(self, x: np.ndarray):
        for act in self.activation_functions.keys():
            if(act == self.activation):
                return self.activation_functions[act](x)

    # We will use X to reffer to the input of the layer
    def calc_output(self, X: np.ndarray):
        self.z = (self.weight @ X) + self.bias
        self.output = self.act_function(self.z)
        return self.output

    def softmax_gradient(self, y_hat: np.ndarray, y:np.ndarray):
        batch_size = y_hat.shape[-1]
        y_real = np.zeros((y_hat.shape))
        y_real[y, np.arange(batch_size)] = 1
        #print(y_hat - y_real)
        return (y_hat - y_real) / batch_size

    def relu_gradient(self, in_gradient: np.array): 
        return (in_gradient.T @ self.nxt_layer_weight).T * (self.z > 0)

    def update_param_softmax(self):
        self.weight = self.weight - self.lr * (self.gradient @ self.prev_layer_out.T)
        self.bias = self.bias - self.lr * np.sum(self.gradient, axis = 1, keepdims = True)

    def update_param_relu(self):
        self.weight = self.weight - self.lr * (self.gradient @ self.prev_layer_out.T)
        self.bias = self.bias - self.lr * np.sum(self.gradient, axis = 1, keepdims = True)

    def calc_gradient(self, in_gradient: np.ndarray, y_hat: np.ndarray = None, y: np.ndarray = None):
        if(self.activation == "softmax"):
            self.gradient = self.softmax_gradient(y_hat, y)
            self.update_param_softmax()
        else:
            self.gradient = self.relu_gradient(in_gradient)
            self.update_param_relu()
