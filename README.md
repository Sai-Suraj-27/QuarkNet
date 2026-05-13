# QuarkNet

[![PyPI](https://img.shields.io/pypi/v/quarknet)](https://pypi.org/project/quarknet/)
[![Python](https://img.shields.io/pypi/pyversions/quarknet)](https://pypi.org/project/quarknet/)
[![License](https://img.shields.io/github/license/Sai-Suraj-27/QuarkNet)](https://github.com/Sai-Suraj-27/QuarkNet/blob/main/LICENSE)

QuarkNet is a NumPy-only neural network library built from scratch for experimenting with fully connected feedforward networks. It includes layer stacking, parameter and gradient tracking, mini-batch training, SGD optimization, and simple `train`/`predict` APIs.

## Features

- Layer stacking with `NeuralNet`
- Fully connected linear layers: `Linear`
- Activation functions: `Tanh`, `ReLU`, `Sigmoid`
- Loss function: mean squared error (`MSE`)
- Optimizer: stochastic gradient descent (`SGD`)
- Mini-batch iteration with optional shuffling: `BatchIterator`
- Installable package published on PyPI

## Installation

Using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv venv .venv
source .venv/bin/activate
uv pip install quarknet
```

Using standard Python tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install quarknet
```

## Quick Start

The XOR problem is non-linear, so it cannot be solved by a single linear layer. This example trains a small network using QuarkNet's training loop.

```python
import numpy as np
from quarknet import Linear, MSE, NeuralNet, SGD, Tanh

np.random.seed(0)

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

targets = np.array([
    [0],
    [1],
    [1],
    [0],
])

model = NeuralNet([
    Linear(2, 2),
    Tanh(),
    Linear(2, 1),
])

model.train(
    inputs=inputs,
    targets=targets,
    loss=MSE(),
    optimizer=SGD(lr=0.1),
    epochs=5000,
    batch_size=4,
    shuffle=True,
    verbose=False,
)

predictions = model.predict(inputs)
print("Predictions:\n", np.round(predictions))
```

Expected output:

```text
Predictions:
 [[0.]
 [1.]
 [1.]
 [0.]]
```

## Examples

- XOR classification: [`examples/xor.ipynb`](examples/xor.ipynb)
- MNIST classification: [`examples/mnist.ipynb`](examples/mnist.ipynb)
- FizzBuzz classification: [`examples/fizz_buzz.ipynb`](examples/fizz_buzz.ipynb)
