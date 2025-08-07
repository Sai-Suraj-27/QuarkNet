"""
Loss functions measure how good are the predictions
made by the neural network. We will use this to tune
the parameters of the neural network.
"""

import numpy as np
from numpy import ndarray


class Loss:
    def loss(self, predictions: ndarray, actual: ndarray) -> float:
        raise NotImplementedError

    def grad(self, predictions: ndarray, actual: ndarray) -> ndarray:
        raise NotImplementedError


class MSE(Loss):
    """
    MSE is the Mean Squared Error.
    """

    def __init__(self) -> None:
        pass

    def loss(self, predictions: ndarray, actual: ndarray) -> float:
        return float(np.mean((predictions - actual) ** 2, axis=None))

    def grad(self, predictions: ndarray, actual: ndarray) -> ndarray:
        return (2 / predictions.size) * (predictions - actual)
