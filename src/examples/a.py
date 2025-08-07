"""

I will have training dataset consisting of ndarray's of features and target

X_train : (n, no_of_features)
y_train : (n, actual_val)

X_test : (t, no_of_features)
y_test : (t, actual_val)


```
uv pip install quarknet
```

```python

from quarknet.layers import Linear
from quarknet.activations import Tanh
from quarknet.nn import NeuralNet
from quarknet.loss import MSE
from quarknet.optim import SGD


model = NeuralNet([
    Linear(512, 64),
    Tanh(),
    Linear(64, 32),
    Tanh(),
    Linear(32, 10),
])

model.train(
    X_train,
    y_train,
    epochs,
    loss=MSE(),
    optimizer=SGD(lr=lr_value)
)


model.predict(
    X_example
)


# Loss should be very less after the training
model.test(
    X_test,
    y_test,
    loss=MSE(),
)

```

"""
