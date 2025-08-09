"""
A NeuralNet is just a collection/stack of Layers.
There are more complicated neural networks
that can not be thought of as a simple stack of layers
but for our library won't handle them.
"""

from typing import Sequence, Iterator
from numpy import ndarray
from .layers import Layer
from .loss import Loss, MSE
from .optim import Optimizer, SGD
from .data import BatchIterator


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: ndarray) -> ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward(self, grad: ndarray) -> ndarray:
        for layer in reversed(self.layers):
            # Populates the grad dict ({}) of all the layers
            grad = layer.backward(grad)

        return grad

    def params_and_grads(self) -> Iterator[tuple[ndarray, ndarray]]:
        for layer in self.layers:
            for parameter_name, parameter_value in layer.params.items():
                gradient = layer.grad[parameter_name]
                yield parameter_value, gradient

    def predict(self, inputs: ndarray) -> ndarray:
        return self.forward(inputs)

    def train(
        self,
        inputs: ndarray,
        targets: ndarray,
        loss: Loss = MSE(),
        optimizer: Optimizer = SGD(),
        epochs: int = 5000,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        for epoch in range(epochs):
            epoch_loss = 0.0
            iterator = BatchIterator(batch_size=batch_size, shuffle=shuffle)

            for batch in iterator(inputs, targets):
                batch_predictions = self.forward(batch.inputs)
                epoch_loss += loss.loss(batch_predictions, batch.targets)
                loss_grad = loss.grad(batch_predictions, batch.targets)
                self.backward(loss_grad)
                optimizer.step(self)

            print(f"Epoch No: {epoch}, Loss: {epoch_loss}")

    def test(
        self,
        inputs: ndarray,
        targets: ndarray,
        loss: Loss = MSE(),
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        total_loss = 0.0
        iterator = BatchIterator(batch_size=batch_size, shuffle=shuffle)

        for batch in iterator(inputs, targets):
            batch_predictions = self.forward(batch.inputs)
            total_loss += loss.loss(batch_predictions, batch.targets)

        print(f"Total Loss: {total_loss}")
