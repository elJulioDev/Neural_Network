import unittest
import numpy as np
import sys
import os

# Ajuste de path para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layer import Layer

class TestLayer(unittest.TestCase):
    
    def setUp(self):
        self.input_size = 3
        self.num_neurons = 4
        self.layer = Layer(self.num_neurons, self.input_size)

    def test_initialization(self):
        """Verifica que se creen la cantidad correcta de neuronas."""
        self.assertEqual(len(self.layer.neurons), self.num_neurons)

    def test_forward_shape(self):
        """La salida debe tener tantos elementos como neuronas."""
        inputs = np.array([1, 2, 3])
        output = self.layer.forward(inputs)
        self.assertEqual(output.shape, (self.num_neurons,))

    def test_backward_output_shape(self):
        """El backward debe devolver gradientes del tamaño de los inputs."""
        inputs = np.array([1.0, 2.0, 3.0])
        self.layer.forward(inputs) # Necesario para guardar inputs
        
        d_outputs = np.ones(self.num_neurons) # Gradiente ficticio
        d_inputs = self.layer.backward(d_outputs, 0.1)
        
        self.assertEqual(d_inputs.shape, (self.input_size,))

if __name__ == '__main__':
    unittest.main()