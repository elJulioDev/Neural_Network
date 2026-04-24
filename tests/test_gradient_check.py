"""
Validación numérica del backprop. Si pasa, la matemática es correcta.

Cubre combinaciones críticas que antes podían fallar silenciosamente:
- Tanh + MSE (capas profundas)
- Softmax + CategoricalCrossEntropy (from_logits=False, Jacobiano real)
- Linear + CategoricalCrossEntropy(from_logits=True) (atajo estable)
"""
import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import NeuralNetwork, Dense, MSE, CategoricalCrossEntropy


def _numeric_gradient(model, X, y, loss_fn, layer_idx, param_name, eps=1e-5):
    """Gradiente numérico por diferencias finitas del parámetro dado."""
    param = model.layers[layer_idx].parameters()[param_name]
    numeric = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]

        param[idx] = orig + eps
        loss_plus = loss_fn.calculate(model.forward(X, training=False), y)

        param[idx] = orig - eps
        loss_minus = loss_fn.calculate(model.forward(X, training=False), y)

        param[idx] = orig
        numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return numeric


def _analytic_gradient(model, X, y, loss_fn, layer_idx, param_name):
    """Gradiente analítico vía backward pass."""
    out, caches = model._forward(X, training=False)
    grad_out = loss_fn.derivative(out, y)
    grads_per_layer = model._backward(grad_out, caches)
    return grads_per_layer[layer_idx][1][param_name]


class TestGradientCheck(unittest.TestCase):

    def test_tanh_mse(self):
        np.random.seed(42)
        model = NeuralNetwork()
        model.add(Dense(4, input_size=3, activation="tanh"))
        model.add(Dense(2, activation="linear"))
        model.compile(optimizer="sgd", loss="mse")

        X = np.random.randn(5, 3)
        y = np.random.randn(5, 2)
        loss = MSE()

        for i in range(2):
            for name in ["weights", "biases"]:
                analytic = _analytic_gradient(model, X, y, loss, i, name)
                numeric = _numeric_gradient(model, X, y, loss, i, name)
                np.testing.assert_allclose(
                    analytic, numeric, atol=1e-4, rtol=1e-3,
                    err_msg=f"Gradient check falló en capa {i} / {name}",
                )

    def test_softmax_cce_full_jacobian_path(self):
        """Softmax + CCE(from_logits=False): usa el Jacobiano real de
        Softmax. Esta combinación se rompía en v0.3 por el acoplamiento."""
        np.random.seed(0)
        model = NeuralNetwork()
        model.add(Dense(3, input_size=2, activation="tanh"))
        model.add(Dense(3, activation="softmax"))
        model.compile(optimizer="sgd", loss=CategoricalCrossEntropy(from_logits=False))

        X = np.random.randn(4, 2) * 0.3
        y = np.eye(3)[np.array([0, 1, 2, 0])]
        loss = CategoricalCrossEntropy(from_logits=False)

        for i, name in [(0, "weights"), (1, "weights")]:
            analytic = _analytic_gradient(model, X, y, loss, i, name)
            numeric = _numeric_gradient(model, X, y, loss, i, name)
            np.testing.assert_allclose(
                analytic, numeric, atol=1e-3, rtol=1e-2,
                err_msg=f"Softmax+CCE full Jacobian gradient check falló en capa {i}",
            )

    def test_linear_cce_from_logits_shortcut(self):
        """Linear + CCE(from_logits=True): atajo numéricamente estable."""
        np.random.seed(0)
        model = NeuralNetwork()
        model.add(Dense(4, input_size=2, activation="relu"))
        model.add(Dense(3, activation="linear"))
        model.compile(optimizer="sgd", loss=CategoricalCrossEntropy(from_logits=True))

        X = np.random.randn(4, 2)
        y = np.eye(3)[np.array([0, 1, 2, 0])]
        loss = CategoricalCrossEntropy(from_logits=True)

        for i in range(2):
            for name in ["weights", "biases"]:
                analytic = _analytic_gradient(model, X, y, loss, i, name)
                numeric = _numeric_gradient(model, X, y, loss, i, name)
                np.testing.assert_allclose(
                    analytic, numeric, atol=1e-4, rtol=1e-3,
                    err_msg=f"from_logits gradient check falló en capa {i} / {name}",
                )


if __name__ == "__main__":
    unittest.main()