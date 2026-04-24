import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layer import Layer
from src.activations import Sigmoid

class TestLayer(unittest.TestCase):
    
    def setUp(self):
        self.input_size = 3
        self.num_neurons = 4
        # Ahora pasamos la activación en el init
        self.layer = Layer(self.num_neurons, self.input_size, activation=Sigmoid())

    def test_initialization(self):
        """Verifica que la matriz de pesos tenga la forma correcta (Input x Neurons)."""
        # Ya no contamos neuronas en una lista, miramos la matriz W
        self.assertEqual(self.layer.weights.shape, (self.input_size, self.num_neurons))
        # Los bias deben ser (1, Neurons)
        self.assertEqual(self.layer.biases.shape, (1, self.num_neurons))

    def test_forward_shape(self):
        """La salida debe ser (Batch_Size x Num_Neurons)."""
        # Simulamos un batch de 5 ejemplos
        batch_size = 5
        inputs = np.random.randn(batch_size, self.input_size)
        
        output = self.layer.forward(inputs)
        self.assertEqual(output.shape, (batch_size, self.num_neurons))

    def test_backward_pass(self):
        """El backward debe devolver gradientes para la capa ANTERIOR (d_inputs)."""
        batch_size = 1
        inputs = np.array([[1.0, 2.0, 3.0]]) # Shape (1, 3)
        self.layer.forward(inputs) # Forward pass para guardar cache
        
        # Simulamos gradiente que viene de la capa siguiente
        d_output = np.ones((batch_size, self.num_neurons))
        
        # IMPORTANTE: Ya no pasamos learning_rate aquí. 
        # Layer.backward solo calcula gradientes.
        d_input = self.layer.backward(d_output)
        
        # Debe devolver gradiente del tamaño de los inputs original
        self.assertEqual(d_input.shape, inputs.shape)
        
        # Verificamos que se calcularon los gradientes de pesos y bias
        self.assertIsNotNone(self.layer.dweights)
        self.assertIsNotNone(self.layer.dbiases)
        self.assertEqual(self.layer.dweights.shape, self.layer.weights.shape)

if __name__ == '__main__':
    unittest.main()