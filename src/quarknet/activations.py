"""
Activations apply a function elementwise to the inputs
"""

from typing import Callable
import numpy as np
from numpy import ndarray
from quarknet.layers import Layer


F = Callable[[ndarray], ndarray]


class Activation(Layer):
    """
    Activation Layers just apply a function elementwise to it's inputs
    """

    def __init__(self, f: F, f_grad: F) -> None:
        self.f = f
        self.f_grad = f_grad

    def forward(self, inputs: ndarray) -> ndarray:
        """ """
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: ndarray) -> ndarray:
        """ """

        return grad * self.f_grad(self.inputs)


def tanh(inputs: ndarray) -> ndarray:
    return np.tanh(inputs)


def tanh_grad(inputs: ndarray) -> ndarray:
    return 1 - (tanh(inputs)) ** 2


class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(tanh, tanh_grad)


def relu(inputs: ndarray) -> ndarray:
    return np.where(inputs >= 0, inputs, 0)


def relu_grad(inputs: ndarray) -> ndarray:
    return np.where(inputs > 0, 1, 0)


class ReLU(Activation):
    def __init__(self) -> None:
        super().__init__(relu, relu_grad)
