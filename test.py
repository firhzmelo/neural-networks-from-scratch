from classes.MyNeuralNetwork import MyNeuralNetworkClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

mnist = pd.read_csv("data/mnist-train.csv")
X = mnist.drop(columns = "label").copy()
y = mnist["label"].copy()
X = X / 255

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123, test_size = 0.2)

def create_batches(X, y, size = 128):
    if X.shape[0] != y.shape[0]:
        raise Exception("Num. de muestras no concuerdan en X y y")
    ini = 0
    samples = X.shape[0]
    batches = [(X[i:i+size], y[i:i+size]) for i in range(0, samples, size)]
    return batches

training_data = create_batches(X_train, y_train)

nn = MyNeuralNetworkClassifier([100], X.shape[-1], classes = 10)

for (batch_x, batch_y) in training_data:
    nn.fit(batch_x, batch_y)
