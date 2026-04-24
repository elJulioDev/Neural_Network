import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimizers import SGD, AdaGrad, RMSprop, Adam, get_optimizer


def make_step(opt, param, grad, layer_id=1, name="w"):
    """Helper: aplica un step del optimizer sobre (param, grad)."""
    opt.apply_gradients([(layer_id, name, param, grad)])


class TestSGD(unittest.TestCase):

    def test_basic_update(self):
        opt = SGD(learning_rate=0.1, momentum=0.0)
        w = np.array([10.0])
        make_step(opt, w, np.array([1.0]))
        self.assertAlmostEqual(w[0], 9.9)

    def test_momentum_accumulates(self):
        opt = SGD(learning_rate=0.1, momentum=0.9)
        w = np.array([10.0])
        make_step(opt, w, np.array([1.0]))
        self.assertAlmostEqual(w[0], 9.9)
        make_step(opt, w, np.array([1.0]))
        self.assertAlmostEqual(w[0], 9.71)


class TestAdam(unittest.TestCase):

    def test_converges_on_quadratic(self):
        opt = Adam(learning_rate=0.1)
        w = np.array([5.0])
        for _ in range(200):
            g = 2.0 * w
            make_step(opt, w, g)
        self.assertLess(abs(w[0]), 0.5)

    def test_first_iter_moves_weights(self):
        opt = Adam(learning_rate=0.001)
        w = np.array([10.0])
        make_step(opt, w, np.array([1.0]))
        self.assertNotEqual(w[0], 10.0)


class TestRMSprop(unittest.TestCase):

    def test_update_moves_weights(self):
        opt = RMSprop(learning_rate=0.001)
        w = np.array([10.0])
        make_step(opt, w, np.array([1.0]))
        self.assertNotEqual(w[0], 10.0)


class TestAdaGrad(unittest.TestCase):

    def test_effective_lr_decays(self):
        opt = AdaGrad(learning_rate=0.1)
        w = np.array([10.0])
        w_init = w[0]
        make_step(opt, w, np.array([1.0]))
        first_step = abs(w[0] - w_init)
        second_init = w[0]
        make_step(opt, w, np.array([1.0]))
        second_step = abs(w[0] - second_init)
        self.assertLess(second_step, first_step)


class TestGradientClipping(unittest.TestCase):

    def test_clip_value(self):
        opt = SGD(learning_rate=1.0, clip_value=0.5)
        w = np.array([0.0])
        make_step(opt, w, np.array([10.0]))
        self.assertAlmostEqual(w[0], -0.5)

    def test_clip_norm(self):
        opt = SGD(learning_rate=1.0, clip_norm=1.0)
        w = np.array([0.0, 0.0])
        make_step(opt, w, np.array([3.0, 4.0]))
        np.testing.assert_allclose(w, [-0.6, -0.8], atol=1e-6)


class TestGenericInterface(unittest.TestCase):
    """Verifica que optimizers no dependan de nombres específicos."""

    def test_handles_arbitrary_param_names(self):
        opt = SGD(learning_rate=0.1)
        gamma = np.array([1.0, 1.0])
        query = np.array([5.0])

        opt.apply_gradients([
            (0, "gamma", gamma, np.array([0.5, 0.5])),
            (0, "query", query, np.array([1.0])),
            (1, "random_name", np.array([0.0]), np.array([2.0])),
        ])
        np.testing.assert_allclose(gamma, [0.95, 0.95])
        self.assertAlmostEqual(query[0], 4.9)


class TestOptimizerConfig(unittest.TestCase):

    def test_adam_roundtrip(self):
        opt = Adam(learning_rate=0.01, beta_1=0.85, clip_norm=1.0)
        cfg = opt.get_config()
        self.assertEqual(cfg["class_name"], "Adam")
        restored = get_optimizer(cfg)
        self.assertEqual(restored.lr, 0.01)
        self.assertEqual(restored.beta_1, 0.85)
        self.assertEqual(restored.clip_norm, 1.0)

    def test_get_optimizer_string(self):
        self.assertIsInstance(get_optimizer("adam"), Adam)
        with self.assertRaises(ValueError):
            get_optimizer("foo")


if __name__ == "__main__":
    unittest.main()