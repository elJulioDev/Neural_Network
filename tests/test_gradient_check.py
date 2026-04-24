"""
Verificación numérica del backward pass.
Compara gradiente analítico contra gradiente por diferencias finitas.
Si esto pasa, la matemática del modelo es correcta.
"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import NeuralNetwork, Dense, MSE


class TestGradientCheck(unittest.TestCase):
    """Si esta prueba pasa, el backprop es matemáticamente correcto."""

    def test_dense_gradient_check(self):
        np.random.seed(42)

        # Red pequeña para checkeo numérico
        model = NeuralNetwork()
        model.add(Dense(4, input_size=3, activation="tanh"))
        model.add(Dense(2, activation="linear"))
        model.compile(optimizer="sgd", loss="mse")

        X = np.random.randn(5, 3)
        y = np.random.randn(5, 2)
        loss_fn = MSE()

        # 1. Gradiente analítico (vía backward)
        out = model.forward(X, training=True)
        grad_out = loss_fn.derivative(out, y)
        model.backward(grad_out)
        analytic = model.layers[0].dweights.copy()

        # 2. Gradiente numérico
        eps = 1e-5
        weights = model.layers[0].weights
        numeric = np.zeros_like(weights)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                orig = weights[i, j]

                weights[i, j] = orig + eps
                loss_plus = loss_fn.calculate(model.forward(X, training=False), y)

                weights[i, j] = orig - eps
                loss_minus = loss_fn.calculate(model.forward(X, training=False), y)

                weights[i, j] = orig
                numeric[i, j] = (loss_plus - loss_minus) / (2 * eps)

        # Comparar (tolerancia razonable por aritmética float)
        np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()