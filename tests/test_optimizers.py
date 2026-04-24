import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layer import Layer
from src.optimizers import SGD
from src.activations import Linear # O usa Sigmoid si no tienes Linear

# Mock simple para probar
class MockLayer:
    def __init__(self):
        self.weights = np.array([10.0])
        self.biases = np.array([0.0])
        self.dweights = np.array([1.0]) # Gradiente simulado
        self.dbiases = np.array([0.5])

class TestOptimizers(unittest.TestCase):
    
    def test_sgd_update(self):
        optimizer = SGD(learning_rate=0.1, momentum=0.0)
        layer = MockLayer()
        
        optimizer.update([layer])
        
        # W_new = W_old - lr * grad
        # 10.0 - 0.1 * 1.0 = 9.9
        self.assertAlmostEqual(layer.weights[0], 9.9)
        
    def test_momentum(self):
        # Con momentum, la actualización depende de la historia
        optimizer = SGD(learning_rate=0.1, momentum=0.9)
        layer = MockLayer()
        
        # Primer paso (igual que SGD normal porque velocidad inicial es 0)
        optimizer.update([layer])
        # Vel_1 = 0.9*0 - 0.1*1.0 = -0.1
        # W_1 = 10.0 + (-0.1) = 9.9
        self.assertAlmostEqual(layer.weights[0], 9.9)
        
        # Segundo paso (el gradiente sigue siendo 1.0)
        optimizer.update([layer])
        # Vel_2 = 0.9*(-0.1) - 0.1*1.0 = -0.09 - 0.1 = -0.19
        # W_2 = 9.9 + (-0.19) = 9.71
        self.assertAlmostEqual(layer.weights[0], 9.71)

if __name__ == '__main__':
    unittest.main()