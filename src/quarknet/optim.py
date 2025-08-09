"""
Optimizer is used to modify the parameters of
our Neural Network based on the gradients computed
during backpropogation.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quarknet.nn import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for parameter_value, gradient in net.params_and_grads():
            parameter_value -= self.lr * gradient
