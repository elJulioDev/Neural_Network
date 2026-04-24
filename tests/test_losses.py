import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.losses import (
    MSE, MAE, Huber,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    SparseCategoricalCrossEntropy,
    get_loss,
)
from src.metrics import (
    BinaryAccuracy, CategoricalAccuracy,
    SparseCategoricalAccuracy, R2Score,
    get_metric,
)


class TestLosses(unittest.TestCase):

    def test_mse(self):
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.5], [0.5]])
        self.assertAlmostEqual(MSE().calculate(y_pred, y_true), 0.25)

    def test_mae(self):
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.7], [0.3]])
        self.assertAlmostEqual(MAE().calculate(y_pred, y_true), 0.3, places=5)

    def test_huber_zero_for_perfect_prediction(self):
        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.0], [2.0]])
        self.assertAlmostEqual(Huber().calculate(y_pred, y_true), 0.0)

    def test_bce_penalizes_wrong_predictions(self):
        bce = BinaryCrossEntropy()
        y_true = np.array([[1.0]])
        loss_good = bce.calculate(np.array([[0.9]]), y_true)
        loss_bad = bce.calculate(np.array([[0.1]]), y_true)
        self.assertLess(loss_good, loss_bad)

    def test_categorical_crossentropy(self):
        cce = CategoricalCrossEntropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        loss = cce.calculate(y_pred, y_true)
        self.assertGreater(loss, 0.0)

    def test_sparse_categorical_crossentropy(self):
        scce = SparseCategoricalCrossEntropy()
        y_true = np.array([0, 1])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        loss = scce.calculate(y_pred, y_true)
        self.assertGreater(loss, 0.0)

        # Gradiente shape correcto
        grad = scce.derivative(y_pred, y_true)
        self.assertEqual(grad.shape, y_pred.shape)

    def test_get_loss_string(self):
        self.assertIsInstance(get_loss("mse"), MSE)
        self.assertIsInstance(get_loss("bce"), BinaryCrossEntropy)
        with self.assertRaises(ValueError):
            get_loss("not_a_loss")


class TestMetrics(unittest.TestCase):

    def test_binary_accuracy(self):
        m = BinaryAccuracy()
        y_pred = np.array([[0.9], [0.1], [0.6], [0.3]])
        y_true = np.array([[1], [0], [1], [1]])
        # 3 correctos de 4
        self.assertAlmostEqual(m(y_pred, y_true), 0.75)

    def test_categorical_accuracy(self):
        m = CategoricalAccuracy()
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
        y_true = np.array([[1, 0], [0, 1]])
        self.assertAlmostEqual(m(y_pred, y_true), 1.0)

    def test_sparse_categorical_accuracy(self):
        m = SparseCategoricalAccuracy()
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
        y_true = np.array([0, 1])
        self.assertAlmostEqual(m(y_pred, y_true), 1.0)

    def test_r2_score_perfect(self):
        m = R2Score()
        y = np.array([[1.0], [2.0], [3.0]])
        self.assertAlmostEqual(m(y, y), 1.0, places=5)

    def test_get_metric(self):
        self.assertIsInstance(get_metric("accuracy"), BinaryAccuracy)
        self.assertIsInstance(get_metric("r2"), R2Score)


if __name__ == "__main__":
    unittest.main()