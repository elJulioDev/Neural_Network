import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.layer import Layer, Dense, Dropout, BatchNormalization, layer_from_config
from src.activations import Sigmoid, ReLU
from src.regularizers import L2


class TestLayerForwardBackward(unittest.TestCase):

    def setUp(self):
        self.layer = Layer(4, 3, activation=Sigmoid())

    def test_initialization(self):
        self.assertEqual(self.layer.weights.shape, (3, 4))
        self.assertEqual(self.layer.biases.shape, (1, 4))

    def test_forward_returns_output_and_cache(self):
        inputs = np.random.randn(5, 3)
        output, cache = self.layer.forward(inputs)
        self.assertEqual(output.shape, (5, 4))
        self.assertIn("inputs", cache)
        self.assertIn("z", cache)
        self.assertIn("activation_cache", cache)

    def test_backward_uses_cache_not_self_state(self):
        inputs = np.array([[1.0, 2.0, 3.0]])
        output, cache = self.layer.forward(inputs)
        d_input, grads = self.layer.backward(np.ones((1, 4)), cache)
        self.assertEqual(d_input.shape, inputs.shape)
        self.assertIn("weights", grads)
        self.assertIn("biases", grads)
        self.assertEqual(grads["weights"].shape, self.layer.weights.shape)

    def test_state_isolation_siamese_pattern(self):
        """Una misma capa debe procesar dos inputs distintos sin
        contaminar el cache."""
        x1 = np.random.randn(3, 3)
        x2 = np.random.randn(3, 3)

        out1, cache1 = self.layer.forward(x1)
        out2, cache2 = self.layer.forward(x2)

        # Backward del primer forward DESPUÉS del segundo forward —
        # antes rompía porque cache2 había pisoteado cache1.
        d1, grads1 = self.layer.backward(np.ones_like(out1), cache1)
        d2, grads2 = self.layer.backward(np.ones_like(out2), cache2)

        self.assertEqual(d1.shape, x1.shape)
        self.assertEqual(d2.shape, x2.shape)
        # Los gradientes deben ser distintos porque los inputs son distintos
        self.assertFalse(np.allclose(grads1["weights"], grads2["weights"]))

    def test_parameters_returns_dict_of_references(self):
        params = self.layer.parameters()
        self.assertIn("weights", params)
        self.assertIn("biases", params)
        # Modificar la ref debe afectar al atributo de la capa
        params["weights"][...] = 0.0
        self.assertTrue(np.all(self.layer.weights == 0.0))

    def test_build_with_input_shape(self):
        l = Layer(4, activation="relu")
        self.assertIsNone(l.weights)
        out_shape = l.build((None, 7))
        self.assertEqual(l.weights.shape, (7, 4))
        self.assertEqual(out_shape, (None, 4))

    def test_shape_mismatch_fail_fast(self):
        inputs = np.random.randn(5, 10)  # esperaba 3 features
        with self.assertRaises(ValueError):
            self.layer.forward(inputs)

    def test_get_config_roundtrip(self):
        l = Layer(8, 4, activation="relu", kernel_regularizer=L2(0.01))
        cfg = l.get_config()
        l2 = layer_from_config(cfg)
        self.assertEqual(l2.n_neurons, 8)
        self.assertIsInstance(l2.activation, ReLU)

    def test_regularization_adds_to_weight_gradient(self):
        l = Layer(4, 3, activation="relu", kernel_regularizer=L2(1.0))
        inputs = np.ones((2, 3))
        _, cache = l.forward(inputs)
        _, grads_no_reg = Layer(4, 3, activation="relu").forward(inputs), None
        _, grads = l.backward(np.ones((2, 4)), cache)
        # El gradiente de weights debe incluir 1.0 * self.weights
        self.assertTrue(np.any(grads["weights"] != 0))


class TestDropout(unittest.TestCase):

    def test_inference_is_identity(self):
        d = Dropout(0.5)
        x = np.random.randn(10, 5)
        out, cache = d.forward(x, training=False)
        np.testing.assert_array_equal(out, x)

    def test_training_drops_elements(self):
        np.random.seed(0)
        d = Dropout(0.5)
        x = np.ones((100, 100))
        out, cache = d.forward(x, training=True)
        zero_ratio = np.mean(out == 0)
        self.assertTrue(0.4 < zero_ratio < 0.6)

    def test_zero_rate_passthrough(self):
        d = Dropout(0.0)
        x = np.random.randn(5, 3)
        out, _ = d.forward(x, training=True)
        np.testing.assert_array_equal(out, x)

    def test_no_trainable_params(self):
        d = Dropout(0.3)
        self.assertEqual(d.parameters(), {})

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            Dropout(1.0)
        with self.assertRaises(ValueError):
            Dropout(-0.1)


class TestBatchNormalization(unittest.TestCase):

    def test_normalizes_during_training(self):
        bn = BatchNormalization(5)
        x = np.random.randn(200, 5) * 10 + 5
        out, _ = bn.forward(x, training=True)
        np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-2)

    def test_parameters_and_state(self):
        bn = BatchNormalization(5)
        params = bn.parameters()
        self.assertIn("gamma", params)
        self.assertIn("beta", params)

        state = bn.non_trainable_state()
        self.assertIn("running_mean", state)
        self.assertIn("running_var", state)

    def test_backward_returns_named_grads(self):
        bn = BatchNormalization(4)
        x = np.random.randn(10, 4)
        _, cache = bn.forward(x, training=True)
        d_in, grads = bn.backward(np.ones((10, 4)), cache)
        self.assertEqual(d_in.shape, x.shape)
        self.assertIn("gamma", grads)
        self.assertIn("beta", grads)

    def test_build_infers_features(self):
        bn = BatchNormalization()
        self.assertIsNone(bn.gamma)
        bn.build((None, 8))
        self.assertEqual(bn.gamma.shape, (1, 8))

    def test_inference_uses_running_stats(self):
        bn = BatchNormalization(3)
        # Training: acumula running stats
        for _ in range(50):
            x = np.random.randn(32, 3) * 2 + 1
            bn.forward(x, training=True)
        # Inference con input constante
        x_test = np.ones((10, 3))
        out, _ = bn.forward(x_test, training=False)
        # No debe fallar y debe producir output válido
        self.assertEqual(out.shape, x_test.shape)


if __name__ == "__main__":
    unittest.main()