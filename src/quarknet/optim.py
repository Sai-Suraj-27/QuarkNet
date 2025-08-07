"""
Optimizer is used to modify the parameters of
our Neural Network based on the gradients computed
during backpropogation.
"""

from quarknet.nn import NeuralNet


class SGD:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for parameter_value, gradient in net.params_and_grads():
            parameter_value -= self.lr * gradient
