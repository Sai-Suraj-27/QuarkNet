"""
Neural Network will be made of layers.
Each layer will pass the inputs it received forward
and propogate the gradients coming from it's next layer backward.
"""

import numpy as np
from numpy import ndarray


class Layer:
    def __init__(self) -> None:
        self.params: dict[str, ndarray] = {}
        self.grad: dict[str, ndarray] = {}

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Return the outputs from the layer corresponding to the inputs
        """
        raise NotImplementedError

    def backward(self, grad: ndarray) -> ndarray:
        """
        Backpropogate the gradient coming w.r.t to the output of the layer
        through the layer and calculates the gradient of the loss fn w.r.t
        the inputs of the layer & along the way saves the gradient of loss
        function w.r.t the weights of the layer.
        """
        raise NotImplementedError


class Linear(Layer):
    """
    Linear Layers computes
    output = inputs @ weights + bias
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.zeros(output_size)

    def forward(self, inputs: ndarray) -> ndarray:
        # Save a copy of the inputs, to use them during backpropogation
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: ndarray) -> ndarray:
        # grad is the gradient of output of this layer w.r.t the loss fn.
        # We need to calculate the gradient of the loss fn w.r.t the inputs & weights
        self.grad["inputs"] = grad @ self.params["w"].T
        self.grad["w"] = self.inputs.T @ grad
        self.grad["b"] = np.sum(grad, axis=0)
        return self.grad["inputs"]
