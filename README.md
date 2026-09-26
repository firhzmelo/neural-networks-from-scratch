# Neural Networks from Scratch

![Status: Work in progress](https://img.shields.io/badge/status-work%20in%20progress-orange)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![NumPy](https://img.shields.io/badge/numpy-only-013243?logo=numpy&logoColor=white)

A from-scratch implementation of an Artificial Neural Network (ANN) classifier written with **NumPy only** — no TensorFlow, PyTorch, or scikit-learn in the core. The goal of this repository is to build the network piece by piece, document the theory behind every step, and eventually compare the result against scikit-learn's `MLPClassifier` on real datasets.

## Table of Contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Project Progress](#project-progress)
- [Project Structure](#project-structure)
- [Implementation Notes](#implementation-notes)
- [Theory](#theory)
- [Resources](#resources)

## Overview

The network is a fully connected feedforward classifier, designed around two pieces:

- **`Layer`** — a single layer: its own weight matrix, bias vector, and activation function.
- **`MyNeuralNetworkClassifier`** — assembles an ordered list of layers, propagates inputs forward, scores the result against the targets with cross-entropy loss, and owns the training loop.

Everything is driven by the chain rule and plain matrix multiplication, so the mathematics stays visible instead of hidden behind a framework.

## Quickstart

Requires Python 3.12 and NumPy:

```bash
pip install numpy
```

Run the scratchpad script to build a small network and inspect its output:

```bash
python3 test.py
```

Usage as in `test.py`:

```python
from classes.MyNeuralNetwork import MyNeuralNetworkClassifier
import numpy as np

X = np.array([1, 2, 3, 4, 5])

nn = MyNeuralNetworkClassifier([3, 4])
nn.fit(X, [1])

print(nn.forward(X))
```

`MyNeuralNetworkClassifier(layers_size)` takes the size of each hidden layer, e.g. `[3, 4]` for two hidden layers of 3 and 4 units. The input layer is inferred from the data during `fit`, and a softmax output layer is appended automatically based on the number of classes found in the targets.

## Project Progress

### Completed

- [x] `Layer` class with weight matrix and bias vector initialization
- [x] Activation functions: ReLU, tanh, sigmoid, softmax
- [x] Forward propagation (`Layer.calc_output`, `MyNeuralNetworkClassifier.forward`)
- [x] Cross-entropy loss (`MyNeuralNetworkClassifier.calc_loss`)
- [x] Network assembly: hidden layers from `layers_size`, input layer inferred from the data, softmax output layer sized from the number of classes

### In Progress

- [ ] Backpropagation (`MyNeuralNetworkClassifier.gradient`)
- [ ] Parameter update step inside `MyNeuralNetworkClassifier.fit` — the loop already iterates `max_iter` times and calls `gradient`, but nothing updates the weights yet, and the learning rate `lr` is accepted but not yet used

### Planned

- [ ] Full-batch and mini-batch gradient descent
- [ ] Weight initialization and regularization strategies
- [ ] Model persistence (save/load trained weights)
- [ ] Evaluation on a real dataset with train/test split
- [ ] Comparison against scikit-learn's `MLPClassifier`

## Project Structure

```
my-neural-network/
├── classes/
│   ├── Layer.py            # Single layer: parameters, activations, forward step
│   └── MyNeuralNetwork.py  # Network assembly, forward pass, loss, training loop
├── test.py                 # Scratchpad / smoke test
└── README.md
```

## Implementation Notes

The design leans toward a **modular approach**: rather than separate `HiddenLayer` and `OutputLayer` classes, a single `Layer` class represents any layer. What changes between layers — the activation function — is just a constructor argument, which keeps the code smaller and makes each layer's behavior identical up to its own parameters.

Notable details:

- **Activations as a registry** — `Layer.activation_functions` maps names to functions, so a layer is configured by passing a string (`"relu"`, `"tanh"`, `"sigmoid"`, `"softmax"`) rather than a callable.
- **Layer sizes are derived, not hard-coded** — `fit` replaces the first layer with one sized to the input and appends the softmax output layer, so the same constructor works for any input width and any number of classes.

### API

| Class | Method | Purpose |
| --- | --- | --- |
| `Layer` | `__init__(layer_size_, prev_size_, activation_="relu")` | Builds weight matrix, bias vector, and selects the activation |
| `Layer` | `act_function(x)` | Applies the configured activation by name |
| `Layer` | `calc_output(inputs)` | Computes `activation(W @ inputs + b)` |
| `MyNeuralNetworkClassifier` | `__init__(layers_size, activation="relu", max_iter_=1000, lr=0.01)` | Builds the hidden layers and stores `max_iter` (`lr` is accepted but not yet retained) |
| `MyNeuralNetworkClassifier` | `forward(X)` | Propagates `X` through every layer |
| `MyNeuralNetworkClassifier` | `calc_loss(X, y)` | Mean cross-entropy loss of the network output |
| `MyNeuralNetworkClassifier` | `gradient(X, y)` | Backpropagation pass (not implemented yet) |
| `MyNeuralNetworkClassifier` | `fit(X, y)` | Sizes the input/output layers and runs the training loop |

## Theory

<!-- TODO: write the theoretical foundations of ANNs here (this section is a placeholder). -->

Suggested sub-topics to cover:

- From the perceptron to the multilayer perceptron
- Forward propagation as a composition of affine transformations and nonlinearities
- Backpropagation and the chain rule
- Choosing activation functions and their derivatives
- Cross-entropy loss and why softmax pairs well with it
- Gradient descent variants (batch, stochastic, mini-batch)

## Resources

- [Stanford CS229: Machine Learning (Autumn 2018) — Lecture 10](https://www.youtube.com/watch?v=MfIjxPh6Pys&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=11)
- [Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math) — Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU)
