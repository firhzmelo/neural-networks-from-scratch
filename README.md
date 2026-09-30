# Neural Networks from Scratch

![Status: Work in progress](https://img.shields.io/badge/status-work%20in%20progress-orange)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![NumPy](https://img.shields.io/badge/numpy-only-013243?logo=numpy&logoColor=white)

A from-scratch implementation of an Artificial Neural Network (ANN) classifier whose core is written with **NumPy only** — no TensorFlow, PyTorch, or scikit-learn in the network itself. The goal of this repository is to build the network piece by piece, document the theory behind every step, and eventually compare the result against scikit-learn's `MLPClassifier` on real datasets.

The first real experiment is a 784 → 100 → 20 fully connected network trained on **MNIST** (42,000 labeled digit images).

## Table of Contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Project Progress](#project-progress)
- [Project Structure](#project-structure)
- [Implementation Notes](#implementation-notes)
- [Known Issues in the Current Code](#known-issues-in-the-current-code)
- [Theory](#theory)
- [Resources](#resources)

## Overview

The network is a fully connected feedforward classifier, designed around two pieces:

- **`Layer`** — a single layer: its own weight matrix, bias vector, and activation function.
- **`MyNeuralNetworkClassifier`** — assembles an ordered list of layers, propagates inputs forward, scores the result against the targets with cross-entropy loss, and runs the training loop.

Everything is driven by the chain rule and plain matrix multiplication, so the mathematics stays visible instead of hidden behind a framework.

## Quickstart

Requires Python 3.12. NumPy is the only dependency of the network; the MNIST experiment additionally uses pandas, scikit-learn and matplotlib:

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Data

`data/mnist-train.csv` holds 42,000 samples with 785 columns: a `label` column plus `pixel0`…`pixel783` with values in `[0, 255]`. The pipeline is:

1. Drop `label` to get `X`, keep it as `y`.
2. Scale the features to `[0, 1]` with `X = X / 255`.
3. Split 80/20 with `train_test_split(..., random_state=123, test_size=0.2)` → 33,600 training samples, 8,400 test samples.
4. Cut the training split into batches of 128 samples.

### Running the experiment

```bash
python3 test.py
```

`main.ipynb` contains the same experiment step by step: load the data, split it, build the batches, train, plot the loss curve and predict a single digit.

Usage, mirroring `test.py`:

```python
from classes.MyNeuralNetwork import MyNeuralNetworkClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

mnist = pd.read_csv("data/mnist-train.csv")
X = mnist.drop(columns="label").copy() / 255
y = mnist["label"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=123, test_size=0.2
)

def create_batches(X, y, size=128):
    if X.shape[0] != y.shape[0]:
        raise Exception("Num. de muestras no concuerdan en X y y")
    return [(X[i:i+size], y[i:i+size]) for i in range(0, X.shape[0], size)]

nn = MyNeuralNetworkClassifier([100], X.shape[-1], classes=10)

for batch_x, batch_y in create_batches(X_train, y_train):
    nn.fit(batch_x, batch_y)

print(nn.predict(X_test.iloc[[0]]))
```

`MyNeuralNetworkClassifier(layers_size, input_size, classes, class_names=[], max_iter=20, lr=0.01, batch_size=128)` builds a network with one ReLU hidden layer per size in `layers_size` and appends a softmax output layer of `classes` units. `fit` and `predict` expect `X` in the usual `(samples, features)` orientation and transpose it to `(features, samples)` internally, since that is how the forward pass multiplies `W @ X`. `fit` performs `max_iter` full-batch updates on whatever matrix it receives, so the caller owns the batching and the number of epochs; pass `return_loss=True` to get the final cross-entropy loss back instead of `None`.

## Project Progress

### Completed

- [x] `Layer` class with He-initialized weight matrix and zero bias vector
- [x] Activation functions: ReLU (hidden default) and numerically stable softmax (output layer only)
- [x] Forward propagation (`Layer.calc_output`, `MyNeuralNetworkClassifier.forward`)
- [x] Cross-entropy loss over the softmax output (`MyNeuralNetworkClassifier.calc_loss`)
- [x] Network assembly: hidden layers from `layers_size`, softmax output layer of `classes` units, sizes wired from `input_size`
- [x] Backpropagation (`MyNeuralNetworkClassifier.backward`, `Layer.calc_gradient`)
- [x] Gradients: fused softmax + cross-entropy on the output layer, masked ReLU gradient on the hidden layers, bias gradients
- [x] SGD parameter update (`Layer.update_param_softmax` / `Layer.update_param_relu`)
- [x] Inference (`MyNeuralNetworkClassifier.predict`)
- [x] MNIST pipeline: CSV loading, `/255` scaling, 80/20 split, batches of 128, per-batch loss tracking
- [x] Training runs end to end on all 263 training batches of the MNIST experiment

### In Progress

- [ ] **No accuracy metric yet** — nothing computes accuracy over `X_test`; the notebook only checks the prediction for one image. Until that exists, the quality of the model is unverified.
- [ ] Loss curve is plotted in the notebook but never saved, and `images/` is still empty.

### Planned

- [ ] Evaluation: accuracy (and per-class breakdown) on the held-out test split
- [ ] Mini-batch training as a feature of the class instead of a caller-side helper
- [ ] Regularization (L2 / dropout)
- [ ] Alternative weight initialization options
- [ ] Model persistence (save/load trained weights)
- [ ] Comparison against scikit-learn's `MLPClassifier`

## Project Structure

```
my-neural-network/
├── classes/
│   ├── Layer.py            # Single layer: parameters, activations, forward/backward steps
│   └── MyNeuralNetwork.py  # Network assembly, forward/backward passes, loss, training loop
├── data/
│   └── mnist-train.csv     # MNIST digits: 42,000 rows x (label + 784 pixels)
├── images/                 # Generated figures (currently empty)
├── main.ipynb              # Step-by-step MNIST experiment: data, training, loss curve, prediction
├── test.py                 # MNIST training loop as a plain script
└── README.md
```

`classes/__pycache__/` is generated by Python and omitted from the tree above.

## Implementation Notes

The design leans toward a **modular approach**: rather than separate `HiddenLayer` and `OutputLayer` classes, a single `Layer` class represents any layer. What changes between layers — the activation function — is just a constructor argument, which keeps the code smaller and makes each layer's behavior identical up to its own parameters.

Notable details:

- **Activations as a registry** — `Layer.activation_functions` maps names to functions, so a layer is configured by passing a string (`"relu"` or `"softmax"`) rather than a callable. Hidden layers default to ReLU; only the output layer uses softmax.
- **He initialization** — weights are drawn from `np.random.randn(size, prev_size) / np.sqrt(prev_size / 2)`, i.e. standard deviation `sqrt(2 / prev_size)`, which keeps activations from collapsing or exploding through ReLU layers. Biases start at zero.
- **Numerically stable softmax** — the maximum is subtracted before exponentiating (`exp(x - max) / sum(exp(x - max))`), so large logits cannot overflow. Outputs are proper probabilities that sum to 1 across the class axis.
- **Fused softmax + cross-entropy gradient** — the output layer uses `(y_hat - onehot(y)) / batch_size` directly instead of multiplying by the softmax Jacobian, which is mathematically identical for this loss pair and avoids building the `(classes, classes)` matrix per sample.
- **Mean gradients** — every gradient is divided by the batch size, and the loss is a mean over samples, so `lr` is independent of batch size.
- **Gradients are computed and applied in one call** — `Layer.calc_gradient` stores `self.gradient` and immediately updates `weight` and `bias`, so `backward` is really "one SGD step over all layers".
- **Biases are column vectors** — `bias` has shape `(size, 1)` while `weight` has shape `(size, prev_size)`, matching the `(features, samples)` orientation used after transposing the input.
- **Type hints on the activations** — `relu` and `softmax` are annotated `np.ndarray -> np.ndarray`.
- **Naming convention** — `X` always refers to the input of a layer and `out` to the output of its activation function, so `calc_output(X)` and `backward(X, out)` stay symmetrical.

### API

| Class | Method | Purpose |
| --- | --- | --- |
| `Layer` | `__init__(layer_size_, prev_size_, lr, activation="relu")` | Builds the weight matrix `(size, prev_size)` and bias column vector `(size, 1)`, and selects the activation |
| `Layer` | `relu(x)` / `softmax(x)` | Type-hinted activation helpers (`np.ndarray -> np.ndarray`) |
| `Layer` | `act_function(x)` | Applies the configured activation by name |
| `Layer` | `calc_output(X)` | Computes `(W @ X) + b`, caches it as `self.z`, and applies the activation |
| `Layer` | `softmax_gradient(y_hat, y)` | Fused softmax + cross-entropy gradient `(y_hat - onehot(y)) / batch_size` |
| `Layer` | `relu_gradient(in_gradient)` | Backpropagates through the next layer's weights and masks by `z > 0` |
| `Layer` | `update_param_softmax()` / `update_param_relu()` | Applies one SGD step to `weight` and `bias` (identical bodies) |
| `Layer` | `calc_gradient(in_gradient, y_hat=None, y=None)` | Computes the gradient for this layer and applies the update |
| `MyNeuralNetworkClassifier` | `__init__(layers_size, input_size, classes, class_names=[], max_iter=50, lr=0.01, batch_size=128)` | Builds the ReLU hidden layers plus the softmax output layer, and stores the hyperparameters |
| `MyNeuralNetworkClassifier` | `forward(X)` | Propagates `X` through every layer, caching each layer's input |
| `MyNeuralNetworkClassifier` | `backward(y, y_hat)` | Walks the layers in reverse, seeding the output layer with the loss gradient |
| `MyNeuralNetworkClassifier` | `calc_loss(y_hat, y)` | Mean cross-entropy loss of the network output |
| `MyNeuralNetworkClassifier` | `fit(X, y, return_loss=False)` | Transposes `X` and runs `max_iter` forward/backward steps, optionally returning the final loss |
| `MyNeuralNetworkClassifier` | `predict(X)` | Transposes `X` and returns `argmax(axis=0)` of the network output |

## Resources

- [Stanford CS229: Machine Learning (Autumn 2018) — Lecture 10](https://www.youtube.com/watch?v=MfIjxPh6Pys&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=11)
- [Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math) — Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU)
- [Explanation of derivative of Softmax](https://davidbieber.com/snippets/2020-12-12-derivative-of-softmax-and-the-softmax-cross-entropy-loss/)
    - More detailed explanation: [link](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
