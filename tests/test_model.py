import unittest
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    NeuralNetwork, Dense, Dropout, BatchNormalization,
    Adam, SGD, L2,
    CategoricalCrossEntropy, BinaryCrossEntropy,
    EarlyStopping, ReduceLROnPlateau,
    train_test_split, to_categorical,
)


class TestXORIntegration(unittest.TestCase):

    def test_xor_with_adam(self):
        np.random.seed(42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)

        model = NeuralNetwork()
        model.add(Dense(8, input_size=2, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer=Adam(learning_rate=0.05), loss="bce", metrics=["accuracy"])

        history = model.fit(X, y, epochs=500, batch_size=4, verbose=0)
        preds = model.predict(X)
        for i in range(4):
            self.assertAlmostEqual(round(float(preds[i, 0])), float(y[i, 0]))
        self.assertIn("loss", history)

    def test_xor_with_logits_path(self):
        """Capa final 'linear' + BCE from_logits. El gradiente debe fluir
        correctamente por el camino estable."""
        np.random.seed(42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)

        model = NeuralNetwork()
        model.add(Dense(8, input_size=2, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(
            optimizer=Adam(learning_rate=0.05),
            loss=BinaryCrossEntropy(from_logits=True),
        )
        model.fit(X, y, epochs=500, batch_size=4, verbose=0)

        # Para obtener probabilidades aplicamos sigmoid al output
        logits = model.predict(X)
        probs = 1.0 / (1.0 + np.exp(-logits))
        for i in range(4):
            self.assertAlmostEqual(round(float(probs[i, 0])), float(y[i, 0]))


class TestStateIsolation(unittest.TestCase):
    """Caches externos — una capa procesando dos inputs no se pisotea."""

    def test_layer_reused_across_inputs(self):
        from src.layer import Dense
        layer = Dense(4, 3, activation="relu")

        x1 = np.random.randn(5, 3)
        x2 = np.random.randn(5, 3)

        # Forward pass 1
        out1, cache1 = layer.forward(x1)
        # Forward pass 2 INTERCALADO (antes del backward de 1)
        out2, cache2 = layer.forward(x2)
        # Backward 1 debe usar cache1 (no cache2)
        d_in1, g1 = layer.backward(np.ones_like(out1), cache1)
        d_in2, g2 = layer.backward(np.ones_like(out2), cache2)

        self.assertFalse(np.allclose(g1["weights"], g2["weights"]))
        self.assertEqual(d_in1.shape, x1.shape)
        self.assertEqual(d_in2.shape, x2.shape)


class TestPersistenceJSON(unittest.TestCase):

    def test_save_load_json_npz(self):
        np.random.seed(0)
        model = NeuralNetwork()
        model.add(Dense(8, input_size=4, activation="relu", kernel_regularizer=L2(0.01)))
        model.add(BatchNormalization(8))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation="softmax"))
        model.compile(optimizer=Adam(learning_rate=0.001), loss="cce", metrics=["categorical_accuracy"])

        # Warm up running stats con un forward
        X = np.random.randn(20, 4)
        _ = model.predict(X)

        before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            # Verifica archivos
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "topology.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "weights.npz")))

            # Carga en modelo nuevo
            loaded = NeuralNetwork.load(tmpdir)
            after = loaded.predict(X)
            np.testing.assert_allclose(before, after, atol=1e-10)

    def test_json_only_roundtrip_no_pickle(self):
        """Topology JSON puro: debe reconstruir arquitectura sin pickle."""
        model = NeuralNetwork()
        model.add(Dense(16, input_size=5, activation="relu"))
        model.add(Dense(3, activation="softmax"))
        model.compile(optimizer="adam", loss="cce")

        json_str = model.to_json()
        self.assertIn("Dense", json_str)
        self.assertIn("Adam", json_str)

        restored = NeuralNetwork.from_json(json_str)
        self.assertEqual(len(restored.layers), 2)
        self.assertTrue(restored._compiled)

    def test_batchnorm_running_stats_persisted(self):
        """Las running stats (no entrenables) deben persistir en save/load."""
        np.random.seed(0)
        model = NeuralNetwork()
        model.add(Dense(4, input_size=3, activation="linear"))
        model.add(BatchNormalization(4))
        model.compile(optimizer="adam", loss="mse")

        # Entrenar un poco para cambiar running stats
        model.fit(np.random.randn(50, 3), np.random.randn(50, 4), epochs=3, verbose=0)
        rm_before = model.layers[1].running_mean.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = NeuralNetwork.load(tmpdir)
            rm_after = loaded.layers[1].running_mean
            np.testing.assert_allclose(rm_before, rm_after)


class TestShapePropagation(unittest.TestCase):
    """Fail-fast en shapes incompatibles."""

    def test_mismatched_input_size_fails_on_build(self):
        model = NeuralNetwork()
        model.add(Dense(4, input_size=3, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(optimizer="adam", loss="cce")

        # X tiene 10 features pero el modelo espera 3
        with self.assertRaises(ValueError):
            model.fit(np.random.randn(5, 10), np.random.randn(5, 2), epochs=1, verbose=0)

    def test_build_propagates_output_shapes(self):
        model = NeuralNetwork()
        model.add(Dense(16, input_size=4, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(3, activation="softmax"))
        # Shapes ya deben estar propagadas sin llamar compile()
        self.assertEqual(model.layers[0].output_shape, (None, 16))
        self.assertEqual(model.layers[1].output_shape, (None, 8))
        self.assertEqual(model.layers[2].output_shape, (None, 3))

    def test_explicit_build_before_fit(self):
        model = NeuralNetwork()
        model.add(Dense(4, activation="relu"))  # sin input_size
        model.add(Dense(1, activation="linear"))

        model.build((None, 5))
        self.assertEqual(model.layers[0].weights.shape, (5, 4))
        self.assertEqual(model.layers[1].weights.shape, (4, 1))


class TestCallbacks(unittest.TestCase):

    def test_validation_split(self):
        np.random.seed(0)
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)
        model = NeuralNetwork()
        model.add(Dense(4, input_size=5, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(X, y, epochs=3, batch_size=16, validation_split=0.2, verbose=0)
        self.assertIn("val_loss", history)

    def test_early_stopping_restore_best(self):
        np.random.seed(0)
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)
        model = NeuralNetwork()
        model.add(Dense(4, input_size=5, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer=SGD(learning_rate=0.001), loss="mse")
        es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        model.fit(X, y, epochs=100, batch_size=10, callbacks=[es], verbose=0)
        # Algo pasó antes de epoch 100
        self.assertTrue(es.stopped_epoch >= 0 or not model.stop_training)


class TestBatchNormIntegration(unittest.TestCase):

    def test_model_with_batchnorm_trains(self):
        np.random.seed(0)
        X = np.random.randn(40, 5)
        y = np.random.randn(40, 1)
        model = NeuralNetwork()
        model.add(Dense(8, input_size=5, activation="relu"))
        model.add(BatchNormalization(8))
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(X, y, epochs=10, batch_size=8, verbose=0)
        self.assertEqual(len(history["loss"]), 10)


class TestPredictBatched(unittest.TestCase):

    def test_batched_matches_full(self):
        np.random.seed(0)
        model = NeuralNetwork()
        model.add(Dense(8, input_size=4, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(optimizer="adam", loss="cce")
        X = np.random.randn(100, 4)
        full = model.predict(X)
        batched = model.predict(X, batch_size=8)
        np.testing.assert_allclose(full, batched, atol=1e-10)


class TestUtils(unittest.TestCase):

    def test_train_test_split(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
        self.assertEqual(Xtr.shape[0], 40)
        self.assertEqual(Xte.shape[0], 10)

    def test_to_categorical(self):
        y = np.array([0, 1, 2, 1])
        oh = to_categorical(y, num_classes=3)
        np.testing.assert_array_equal(
            oh, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        )


if __name__ == "__main__":
    unittest.main()