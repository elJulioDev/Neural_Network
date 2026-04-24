import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.losses import (
    MSE, MAE, Huber,
    BinaryCrossEntropy, CategoricalCrossEntropy, SparseCategoricalCrossEntropy,
    get_loss,
)
from src.metrics import (
    BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy, R2Score,
    get_metric,
)


class TestLosses(unittest.TestCase):

    def test_mse(self):
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.5], [0.5]])
        self.assertAlmostEqual(MSE().calculate(y_pred, y_true), 0.25)

    def test_huber_zero_for_perfect(self):
        y = np.array([[1.0], [2.0]])
        self.assertAlmostEqual(Huber().calculate(y, y), 0.0)

    def test_bce_prob_mode(self):
        bce = BinaryCrossEntropy(from_logits=False)
        y_true = np.array([[1.0]])
        self.assertLess(bce.calculate(np.array([[0.9]]), y_true),
                        bce.calculate(np.array([[0.1]]), y_true))

    def test_bce_from_logits_matches_prob_version(self):
        logits = np.array([[0.5], [-1.2], [2.0]])
        y = np.array([[1.0], [0.0], [1.0]])
        sig = 1.0 / (1.0 + np.exp(-logits))
        l_logits = BinaryCrossEntropy(from_logits=True).calculate(logits, y)
        l_prob = BinaryCrossEntropy(from_logits=False).calculate(sig, y)
        self.assertAlmostEqual(l_logits, l_prob, places=6)

    def test_bce_from_logits_gradient(self):
        logits = np.array([[0.5, -1.0], [2.0, 0.0]])
        y = np.array([[1.0, 0.0], [0.0, 1.0]])
        grad = BinaryCrossEntropy(from_logits=True).derivative(logits, y)
        sig = 1.0 / (1.0 + np.exp(-logits))
        expected = (sig - y) / y.size
        np.testing.assert_allclose(grad, expected)

    def test_cce_from_logits_matches_softmax_version(self):
        logits = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        y = np.array([[0, 0, 1], [1, 0, 0]], dtype=float)
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        sm = ex / ex.sum(axis=1, keepdims=True)
        l_logits = CategoricalCrossEntropy(from_logits=True).calculate(logits, y)
        l_prob = CategoricalCrossEntropy(from_logits=False).calculate(sm, y)
        self.assertAlmostEqual(l_logits, l_prob, places=6)

    def test_cce_from_logits_stable_with_large_values(self):
        logits = np.array([[100.0, 0.0], [0.0, -100.0]])
        y = np.array([[1, 0], [1, 0]], dtype=float)
        loss = CategoricalCrossEntropy(from_logits=True).calculate(logits, y)
        self.assertTrue(np.isfinite(loss))

    def test_sparse_cce(self):
        logits = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.7]])
        y = np.array([2, 0])
        sm_loss = SparseCategoricalCrossEntropy(from_logits=True).calculate(logits, y)
        self.assertGreater(sm_loss, 0.0)

    def test_get_loss_roundtrip(self):
        loss = BinaryCrossEntropy(from_logits=True)
        cfg = loss.get_config()
        restored = get_loss(cfg)
        self.assertTrue(restored.from_logits)

    def test_get_loss_string(self):
        self.assertIsInstance(get_loss("mse"), MSE)
        with self.assertRaises(ValueError):
            get_loss("not_a_loss")


class TestMetrics(unittest.TestCase):

    def test_binary_accuracy(self):
        m = BinaryAccuracy()
        y_pred = np.array([[0.9], [0.1], [0.6], [0.3]])
        y_true = np.array([[1], [0], [1], [1]])
        self.assertAlmostEqual(m(y_pred, y_true), 0.75)

    def test_categorical_accuracy(self):
        m = CategoricalAccuracy()
        self.assertAlmostEqual(m(
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.array([[1, 0], [0, 1]])
        ), 1.0)

    def test_r2_perfect(self):
        y = np.array([[1.0], [2.0], [3.0]])
        self.assertAlmostEqual(R2Score()(y, y), 1.0, places=5)

    def test_metric_config(self):
        m = BinaryAccuracy(threshold=0.7)
        cfg = m.get_config()
        restored = get_metric(cfg)
        self.assertEqual(restored.threshold, 0.7)


if __name__ == "__main__":
    unittest.main()