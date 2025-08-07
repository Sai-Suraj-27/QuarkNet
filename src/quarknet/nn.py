"""
A NeuralNet is just a collection/stack of Layers.
There are more complicated neural networks
that can not be thought of as a simple stack of layers
but for our library won't handle them.
"""

from typing import Sequence, Iterator
import numpy as np
from numpy import ndarray
from quarknet.layers import Layer


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
