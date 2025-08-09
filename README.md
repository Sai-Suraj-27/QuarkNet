## QuarkNet ⚡ 

***A minimal deep learning library built from scratch with just NumPy.***

QuarkNet is a lightweight Deep learning library that's built entirely with NumPy and supports the essentials needed to train fully-connected feedforward neural networks.


### Features

🔸Fully Connected Linear Layers (`Linear`)

🔸Activation Functions: `Tanh`, `ReLU`, `Sigmoid`

🔸Loss Functions: Mean Squared Error (`MSE`)

🔸Optimizer: Stochastic Gradient Descent (`SGD`)

🔸Mini-batch Training with shuffling

🔸Layer Stacking API for building networks

🔸Prediction & Evaluation support

🔸Pure NumPy Implementation — no external ML frameworks / Dependencies


### 🚀 Installation

**Create a virtual environment using [uv](https://docs.astral.sh/uv/getting-started/installation/)**

```bash
uv venv .venv
```

**Activate the virtual environment**

```bash
source .venv/bin/activate
```

**Install QuarkNet into your isolated `.venv` virtual environment**

```bash
uv pip install quarknet
```

### Alternatively, if you prefer traditional tools

```bash
python3 -m venv .venv      # Create a virtual environment
source .venv/bin/activate
pip install QuarkNet        # Standard pip install
```


### Solving the XOR Problem with QuarkNet ⚡

The XOR (exclusive OR) problem is a classic example in neural networks because it is non-linear and cannot be solved by a single linear layer.

```python
import numpy as np
from quarknet.nn import NeuralNet
from quarknet.layers import Linear
from quarknet.activations import Tanh
from quarknet.loss import MSE
from quarknet.optim import SGD

# XOR inputs
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# XOR outputs
targets = np.array([
    [0],
    [1], 
    [1],
    [0],
])

# Train the model
model.train(
    inputs=inputs,
    targets=targets,
    loss=MSE(),
    optimizer=SGD(),
    epochs=5000,
    batch_size=32,
    shuffle=True
)

# Make predictions
predictions = model.predict(inputs)
print("Predictions:\n", np.round(predictions))

```

**Output**

```bash
Predictions:
array(
    [
        [0.],
        [1.],
        [1.],
        [0.],
    ]
)
```
