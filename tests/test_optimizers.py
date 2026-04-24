import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimizers import SGD, AdaGrad, RMSprop, Adam, get_optimizer


class MockLayer:
    """Simula una capa con weights, biases y gradientes."""
    trainable = True

    def __init__(self):
        self.weights = np.array([10.0])
        self.biases = np.array([0.0])
        self.dweights = np.array([1.0])
        self.dbiases = np.array([0.5])


class TestSGD(unittest.TestCase):

    def test_basic_update(self):
        opt = SGD(learning_rate=0.1, momentum=0.0)
        layer = MockLayer()
        opt.update([layer])
        self.assertAlmostEqual(layer.weights[0], 9.9)

    def test_momentum_accumulates(self):
        opt = SGD(learning_rate=0.1, momentum=0.9)
        layer = MockLayer()
        opt.update([layer])
        self.assertAlmostEqual(layer.weights[0], 9.9)
        opt.update([layer])
        # v_2 = 0.9*(-0.1) - 0.1*1.0 = -0.19
        self.assertAlmostEqual(layer.weights[0], 9.71)


class TestAdam(unittest.TestCase):

    def test_first_iteration_with_bias_correction(self):
        opt = Adam(learning_rate=0.001)
        layer = MockLayer()
        old_w = layer.weights[0]
        opt.update([layer])
        # Con bias correction, primera iter ~ -lr * sign(g)
        self.assertNotEqual(layer.weights[0], old_w)
        self.assertLess(layer.weights[0], old_w)

    def test_converges_on_quadratic(self):
        # Probamos que Adam minimice f(w) = w^2 (gradiente = 2w)
        class QuadLayer:
            trainable = True
            weights = np.array([5.0])
            biases = np.array([0.0])
            dweights = np.array([0.0])
            dbiases = np.array([0.0])

        layer = QuadLayer()
        opt = Adam(learning_rate=0.1)
        for _ in range(200):
            layer.dweights = 2.0 * layer.weights
            opt.update([layer])
        # Debería acercarse mucho a cero
        self.assertLess(abs(layer.weights[0]), 0.5)


class TestRMSprop(unittest.TestCase):

    def test_update_changes_weights(self):
        opt = RMSprop(learning_rate=0.001)
        layer = MockLayer()
        old_w = layer.weights[0]
        opt.update([layer])
        self.assertNotEqual(layer.weights[0], old_w)


class TestAdaGrad(unittest.TestCase):

    def test_lr_decreases_effective(self):
        opt = AdaGrad(learning_rate=0.1)
        layer = MockLayer()
        opt.update([layer])
        first_change = 10.0 - layer.weights[0]
        opt.update([layer])
        second_change = (10.0 - first_change) - layer.weights[0]
        # AdaGrad debe hacer pasos más pequeños progresivamente
        self.assertLess(second_change, first_change)


class TestGradientClipping(unittest.TestCase):

    def test_clip_value(self):
        opt = SGD(learning_rate=1.0, clip_value=0.5)

        class BigGradLayer:
            trainable = True
            weights = np.array([0.0])
            biases = np.array([0.0])
            dweights = np.array([10.0])
            dbiases = np.array([0.0])

        layer = BigGradLayer()
        opt.update([layer])
        # Sin clip: w = 0 - 1.0 * 10 = -10. Con clip a 0.5: w = -0.5
        self.assertAlmostEqual(layer.weights[0], -0.5)

    def test_clip_norm(self):
        opt = SGD(learning_rate=1.0, clip_norm=1.0)

        class BigGradLayer:
            trainable = True
            weights = np.array([0.0, 0.0])
            biases = np.array([0.0])
            dweights = np.array([3.0, 4.0])  # norma 5
            dbiases = np.array([0.0])

        layer = BigGradLayer()
        opt.update([layer])
        # Gradiente reescalado a norma 1: [0.6, 0.8]
        np.testing.assert_allclose(layer.weights, [-0.6, -0.8], atol=1e-6)


class TestGetOptimizer(unittest.TestCase):

    def test_strings(self):
        self.assertIsInstance(get_optimizer("adam"), Adam)
        self.assertIsInstance(get_optimizer("sgd"), SGD)
        with self.assertRaises(ValueError):
            get_optimizer("foo")


if __name__ == "__main__":
    unittest.main()