import unittest
import numpy as np
import sys
import os

# Esto asegura que podamos importar desde la carpeta 'src' aunque estemos en 'tests'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neuron import Neuron

class TestNeuron(unittest.TestCase):
    
    def setUp(self):
        """Se ejecuta antes de cada prueba. Crea una neurona nueva."""
        self.input_size = 3
        self.neuron = Neuron(self.input_size)

    def test_initialization(self):
        """Verifica que los pesos y el sesgo se inicien correctamente."""
        self.assertEqual(self.neuron.weights.shape, (self.input_size,))
        self.assertIsInstance(self.neuron.bias, float)
        # Los gradientes deben iniciar en cero o valores por defecto
        self.assertTrue(np.all(self.neuron.dweight == 0))

    def test_activation_sigmoid(self):
        """Verifica la matemática de la función sigmoide."""
        # Sigmoide de 0 debe ser 0.5
        result = self.neuron.activate(0)
        self.assertEqual(result, 0.5)
        # Sigmoide de valores muy altos tiende a 1
        self.assertAlmostEqual(self.neuron.activate(100), 1.0, places=5)
        # Sigmoide de valores muy bajos tiende a 0
        self.assertAlmostEqual(self.neuron.activate(-100), 0.0, places=5)

    def test_forward_pass_output_range(self):
        """El output siempre debe estar entre 0 y 1 debido a la sigmoide."""
        inputs = np.array([10.0, 20.0, 30.0])
        output = self.neuron.forward(inputs)
        self.assertGreaterEqual(output, 0.0)
        self.assertLessEqual(output, 1.0)

    def test_backward_updates_weights(self):
        """Verifica que el backward pass modifique los pesos."""
        inputs = np.array([1.0, 1.0, 1.0])
        
        # Guardamos los pesos originales (hacemos una copia)
        original_weights = self.neuron.weights.copy()
        original_bias = self.neuron.bias
        
        # Ejecutamos forward y backward
        self.neuron.forward(inputs)
        d_output = 0.5 # Simulamos un error
        learning_rate = 0.1
        self.neuron.backward(d_output, learning_rate)
        
        # Verificamos que los pesos hayan cambiado
        # Nota: Es matemáticamente posible pero improbable que no cambien, 
        # pero en la práctica siempre cambian si hay learning_rate y error.
        self.assertFalse(np.array_equal(original_weights, self.neuron.weights), 
                         "Los pesos no se actualizaron después del backward pass")
        self.assertNotEqual(original_bias, self.neuron.bias, 
                            "El sesgo (bias) no se actualizó")

if __name__ == '__main__':
    unittest.main()