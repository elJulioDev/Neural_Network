import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.layer import Layer, Dense, Dropout, BatchNormalization
from src.activations import Sigmoid, ReLU
from src.regularizers import L2


class TestLayer(unittest.TestCase):

    def setUp(self):
        self.input_size = 3
        self.num_neurons = 4
        self.layer = Layer(self.num_neurons, self.input_size, activation=Sigmoid())

    def test_initialization(self):
        self.assertEqual(self.layer.weights.shape, (self.input_size, self.num_neurons))
        self.assertEqual(self.layer.biases.shape, (1, self.num_neurons))

    def test_forward_shape(self):
        batch_size = 5
        inputs = np.random.randn(batch_size, self.input_size)
        output = self.layer.forward(inputs)
        self.assertEqual(output.shape, (batch_size, self.num_neurons))

    def test_backward_pass(self):
        batch_size = 1
        inputs = np.array([[1.0, 2.0, 3.0]])
        self.layer.forward(inputs)
        d_output = np.ones((batch_size, self.num_neurons))
        d_input = self.layer.backward(d_output)
        self.assertEqual(d_input.shape, inputs.shape)
        self.assertEqual(self.layer.dweights.shape, self.layer.weights.shape)

    def test_dense_alias(self):
        d = Dense(8, 4, activation="relu")
        self.assertIsInstance(d, Layer)

    def test_string_activation(self):
        layer = Layer(4, 3, activation="relu")
        out = layer.forward(np.random.randn(2, 3))
        self.assertEqual(out.shape, (2, 4))
        # ReLU debe producir todo >= 0
        self.assertTrue(np.all(out >= 0))

    def test_string_initializer(self):
        layer = Layer(4, 3, activation="relu", kernel_initializer="xavier")
        self.assertEqual(layer.weights.shape, (3, 4))

    def test_regularization(self):
        layer = Layer(4, 3, activation="relu", kernel_regularizer=L2(0.01))
        loss = layer.regularization_loss()
        self.assertGreater(loss, 0.0)
        # El gradiente debe sumarse en backward
        layer.forward(np.random.randn(2, 3))
        layer.backward(np.ones((2, 4)))
        self.assertIsNotNone(layer.dweights)

    def test_get_set_params(self):
        params = self.layer.get_params()
        self.assertEqual(len(params), 2)
        new_params = [np.zeros_like(params[0]), np.ones_like(params[1])]
        self.layer.set_params(new_params)
        np.testing.assert_array_equal(self.layer.weights, new_params[0])
        np.testing.assert_array_equal(self.layer.biases, new_params[1])


class TestDropout(unittest.TestCase):

    def test_inference_is_identity(self):
        d = Dropout(0.5)
        x = np.random.randn(10, 5)
        out = d.forward(x, training=False)
        np.testing.assert_array_equal(out, x)

    def test_training_drops_elements(self):
        np.random.seed(0)
        d = Dropout(0.5)
        x = np.ones((100, 100))
        out = d.forward(x, training=True)
        # ~50% deben ser cero
        zero_ratio = np.mean(out == 0)
        self.assertTrue(0.4 < zero_ratio < 0.6)

    def test_zero_rate(self):
        d = Dropout(0.0)
        x = np.random.randn(5, 3)
        out = d.forward(x, training=True)
        np.testing.assert_array_equal(out, x)

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            Dropout(1.0)
        with self.assertRaises(ValueError):
            Dropout(-0.1)


class TestBatchNormalization(unittest.TestCase):

    def test_forward_shape(self):
        bn = BatchNormalization(5)
        x = np.random.randn(10, 5)
        out = bn.forward(x, training=True)
        self.assertEqual(out.shape, x.shape)

    def test_normalizes_during_training(self):
        bn = BatchNormalization(5)
        x = np.random.randn(100, 5) * 10 + 5
        out = bn.forward(x, training=True)
        # Salida debe tener media ~0 y std ~1 (gamma=1, beta=0)
        np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-2)

    def test_backward_shape(self):
        bn = BatchNormalization(5)
        x = np.random.randn(10, 5)
        bn.forward(x, training=True)
        dx = bn.backward(np.ones((10, 5)))
        self.assertEqual(dx.shape, x.shape)
        self.assertIsNotNone(bn.dgamma)
        self.assertIsNotNone(bn.dbeta)


if __name__ == "__main__":
    unittest.main()