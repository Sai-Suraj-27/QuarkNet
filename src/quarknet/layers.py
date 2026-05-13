"""
Layer abstractions for manual forward and backward passes.
"""

import numpy as np
from numpy import ndarray


class Layer:
    def __init__(self) -> None:
        self.params: dict[str, ndarray] = {}
        self.grad: dict[str, ndarray] = {}

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Return the layer output for the provided inputs.
        """
        raise NotImplementedError

    def backward(self, grad: ndarray) -> ndarray:
        """
        Propagate the loss gradient backward through the layer.

        Implementations return the gradient with respect to the inputs and
        store parameter gradients in ``self.grad``.
        """
        raise NotImplementedError


class Linear(Layer):
    """
    Linear layer that computes ``output = inputs @ weights + bias``.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.zeros(output_size)

    def forward(self, inputs: ndarray) -> ndarray:
        # Keep inputs for the manual backward pass.
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: ndarray) -> ndarray:
        self.grad["w"] = self.inputs.T @ grad
        self.grad["b"] = np.sum(grad, axis=0)
        return grad @ self.params["w"].T
