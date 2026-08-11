import unittest

import numpy as np

from nnlib.activations import (
    ELU,
    LeakyReLU,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    get_activation,
)


class TestActivationsStateless(unittest.TestCase):
    """All activations must be stateless: forward does not mutate self."""

    def test_all_return_cache(self):
        x = np.random.randn(4, 3)
        for act_cls in [Sigmoid, ReLU, LeakyReLU, ELU, Tanh, Softmax, Linear]:
            act = act_cls()
            out, cache = act.forward(x)
            self.assertEqual(out.shape, x.shape)
            self.assertIsInstance(cache, dict)

    def test_reuse_same_instance_different_inputs(self):
        """Siamese networks: same instance, two inputs, both backward
        must produce consistent independent results."""
        act = Sigmoid()
        x1 = np.random.randn(3, 4)
        x2 = np.random.randn(3, 4)
        out1, cache1 = act.forward(x1)
        out2, cache2 = act.forward(x2)
        d1 = act.backward(np.ones_like(out1), cache1)
        act.backward(np.ones_like(out2), cache2)
        # Backward of x1 must use the cache of x1, not that of x2
        expected_d1 = out1 * (1 - out1)
        np.testing.assert_allclose(d1, expected_d1, atol=1e-10)

    def test_softmax_full_jacobian(self):
        """Softmax.backward must implement the full Jacobian
        (not the trick of returning 1 assuming CCE)."""
        sm = Softmax()
        x = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.7]])
        out, cache = sm.forward(x)

        # Numerical verification: each row must sum to 1
        np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0])

        # Jacobian verification: numerical gradient check
        d_output = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        analytic = sm.backward(d_output, cache)

        # Numerical
        eps = 1e-6
        numeric = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] += eps
                out_plus, _ = sm.forward(x)
                x[i, j] -= 2 * eps
                out_minus, _ = sm.forward(x)
                x[i, j] += eps
                # d(sum(d_output * out)) / d(x_ij)
                numeric[i, j] = (
                    np.sum(d_output * out_plus) - np.sum(d_output * out_minus)
                ) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


class TestActivationConfig(unittest.TestCase):

    def test_leaky_relu_roundtrip(self):
        l = LeakyReLU(alpha=0.2)
        cfg = l.get_config()
        self.assertEqual(cfg["config"]["alpha"], 0.2)
        l2 = LeakyReLU.from_config(cfg["config"])
        self.assertEqual(l2.alpha, 0.2)

    def test_get_activation_from_dict(self):
        cfg = {"class_name": "LeakyReLU", "config": {"alpha": 0.3}}
        act = get_activation(cfg)
        self.assertEqual(act.alpha, 0.3)

    def test_get_activation_string(self):
        self.assertIsInstance(get_activation("relu"), ReLU)
        self.assertIsInstance(get_activation("softmax"), Softmax)


if __name__ == "__main__":
    unittest.main()
