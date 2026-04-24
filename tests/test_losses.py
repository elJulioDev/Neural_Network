import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.losses import MSE, BinaryCrossEntropy

class TestLosses(unittest.TestCase):
    
    def test_mse_calculation(self):
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.5, 0.5])
        # MSE = ((1-0.5)^2 + (0-0.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
        mse = MSE()
        self.assertEqual(mse.calculate(y_pred, y_true), 0.25)

    def test_bce_calculation(self):
        # BCE penaliza mucho el error
        y_true = np.array([1.0])
        y_pred_good = np.array([0.9])
        y_pred_bad = np.array([0.1])
        
        bce = BinaryCrossEntropy()
        loss_good = bce.calculate(y_pred_good, y_true)
        loss_bad = bce.calculate(y_pred_bad, y_true)
        
        self.assertLess(loss_good, loss_bad)

if __name__ == '__main__':
    unittest.main()