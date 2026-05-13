"""
Loss functions measure prediction error and expose gradients for training.
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
    Mean squared error loss.
    """

    def __init__(self) -> None:
        pass

    def loss(self, predictions: ndarray, actual: ndarray) -> float:
        return float(np.mean((predictions - actual) ** 2, axis=None))

    def grad(self, predictions: ndarray, actual: ndarray) -> ndarray:
        return (2 / predictions.size) * (predictions - actual)
