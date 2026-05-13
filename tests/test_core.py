import numpy as np

from quarknet import BatchIterator, Linear, MSE, NeuralNet, ReLU, SGD, Tanh


def test_linear_forward_backward_tracks_manual_gradients() -> None:
    layer = Linear(2, 1)
    layer.params["w"] = np.array([[1.0], [2.0]])
    layer.params["b"] = np.array([0.5])

    inputs = np.array([[3.0, 4.0], [1.0, -1.0]])
    outputs = layer.forward(inputs)

    np.testing.assert_allclose(outputs, np.array([[11.5], [-0.5]]))

    upstream_grad = np.array([[1.0], [2.0]])
    input_grad = layer.backward(upstream_grad)

    np.testing.assert_allclose(layer.grad["w"], np.array([[5.0], [2.0]]))
    np.testing.assert_allclose(layer.grad["b"], np.array([3.0]))
    np.testing.assert_allclose(input_grad, np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_activation_backward_uses_elementwise_derivative() -> None:
    activation = ReLU()
    inputs = np.array([[-1.0, 0.0, 2.0]])

    np.testing.assert_allclose(activation.forward(inputs), np.array([[0.0, 0.0, 2.0]]))
    np.testing.assert_allclose(
        activation.backward(np.ones_like(inputs)),
        np.array([[0.0, 0.0, 1.0]]),
    )


def test_mse_loss_and_gradient() -> None:
    loss = MSE()
    predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
    actual = np.array([[1.0, 1.0], [1.0, 1.0]])

    assert loss.loss(predictions, actual) == 3.5
    np.testing.assert_allclose(
        loss.grad(predictions, actual),
        np.array([[0.0, 0.5], [1.0, 1.5]]),
    )


def test_batch_iterator_preserves_order_when_shuffle_is_disabled() -> None:
    inputs = np.arange(10).reshape(5, 2)
    targets = np.arange(5)

    batches = list(BatchIterator(batch_size=2, shuffle=False)(inputs, targets))

    assert len(batches) == 3
    np.testing.assert_array_equal(batches[0].inputs, inputs[:2])
    np.testing.assert_array_equal(batches[0].targets, targets[:2])
    np.testing.assert_array_equal(batches[2].inputs, inputs[4:])
    np.testing.assert_array_equal(batches[2].targets, targets[4:])


def test_neural_net_backward_exposes_params_and_grads() -> None:
    np.random.seed(0)
    model = NeuralNet([Linear(2, 3), Tanh(), Linear(3, 1)])
    inputs = np.array([[0.0, 1.0], [1.0, 0.0]])

    predictions = model.forward(inputs)
    input_grad = model.backward(np.ones_like(predictions))

    assert input_grad.shape == inputs.shape
    for parameter, gradient in model.params_and_grads():
        assert parameter.shape == gradient.shape


def test_sgd_updates_parameters_in_place() -> None:
    layer = Linear(2, 1)
    layer.params["w"] = np.array([[1.0], [2.0]])
    layer.params["b"] = np.array([0.5])

    model = NeuralNet([layer])
    predictions = model.forward(np.array([[3.0, 4.0]]))
    loss_grad = MSE().grad(predictions, np.array([[1.0]]))
    model.backward(loss_grad)

    SGD(lr=0.1).step(model)

    np.testing.assert_allclose(layer.params["w"], np.array([[-5.3], [-6.4]]))
    np.testing.assert_allclose(layer.params["b"], np.array([-1.6]))
