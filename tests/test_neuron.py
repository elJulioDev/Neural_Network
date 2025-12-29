import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neuron import Neuron
from src.activations import Sigmoid # <--- Necesitamos importar esto

class TestNeuron(unittest.TestCase):
    
    def setUp(self):
        """Se ejecuta antes de cada prueba. Crea una neurona con Sigmoid."""
        self.input_size = 3
        # CORRECCIÓN: Ahora debemos pasarle una activación explícita
        self.neuron = Neuron(self.input_size, activation=Sigmoid())

    def test_initialization(self):
        self.assertEqual(self.neuron.weights.shape, (self.input_size,))
        self.assertIsInstance(self.neuron.bias, float)
        self.assertTrue(np.all(self.neuron.dweight == 0))

    def test_activation_sigmoid(self):
        # Esta prueba sigue siendo válida porque le pasamos Sigmoid en el setUp
        result = self.neuron.activate(0)
        self.assertEqual(result, 0.5)
        self.assertAlmostEqual(self.neuron.activate(100), 1.0, places=5)
        self.assertAlmostEqual(self.neuron.activate(-100), 0.0, places=5)

    def test_forward_pass_output_range(self):
        inputs = np.array([10.0, 20.0, 30.0])
        output = self.neuron.forward(inputs)
        self.assertGreaterEqual(output, 0.0)
        self.assertLessEqual(output, 1.0)

    def test_backward_updates_weights(self):
        inputs = np.array([1.0, 1.0, 1.0])
        original_weights = self.neuron.weights.copy()
        original_bias = self.neuron.bias
        
        self.neuron.forward(inputs)
        d_output = 0.5 
        learning_rate = 0.1
        self.neuron.backward(d_output, learning_rate)
        
        self.assertFalse(np.array_equal(original_weights, self.neuron.weights), 
                         "Los pesos no se actualizaron")
        self.assertNotEqual(original_bias, self.neuron.bias, 
                            "El sesgo no se actualizó")

if __name__ == '__main__':
    unittest.main()