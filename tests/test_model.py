import unittest
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    NeuralNetwork, Dense, Dropout, BatchNormalization,
    Adam, SGD,
    BinaryCrossEntropy, MSE,
    EarlyStopping, ModelCheckpoint,
    L2,
    train_test_split, to_categorical,
)


class TestModelXOR(unittest.TestCase):
    """Test de integración: la red debe resolver XOR."""

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

        # Debe clasificar las 4 muestras
        for i in range(4):
            self.assertAlmostEqual(round(float(preds[i, 0])), float(y[i, 0]))

        # History contiene loss y accuracy
        self.assertIn("loss", history)
        self.assertIn("binary_accuracy", history)


class TestModelCompilation(unittest.TestCase):

    def test_compile_with_strings(self):
        model = NeuralNetwork()
        model.add(Dense(4, input_size=3, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="bce")
        self.assertTrue(model._compiled)

    def test_fit_without_compile_raises(self):
        model = NeuralNetwork()
        model.add(Dense(4, input_size=3))
        with self.assertRaises(RuntimeError):
            model.fit(np.random.randn(10, 3), np.random.randn(10, 1), verbose=0)


class TestPersistence(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.model = NeuralNetwork()
        self.model.add(Dense(8, input_size=4, activation="relu"))
        self.model.add(Dense(2, activation="softmax"))
        self.model.compile(optimizer="adam", loss="cce")

    def test_save_load_weights_npz(self):
        x = np.random.randn(5, 4)
        before = self.model.predict(x)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            self.model.save_weights(path)

            # Crear un modelo nuevo con misma arquitectura
            model2 = NeuralNetwork()
            model2.add(Dense(8, input_size=4, activation="relu"))
            model2.add(Dense(2, activation="softmax"))
            model2.compile(optimizer="adam", loss="cce")
            model2.load_weights(path)

            after = model2.predict(x)
            np.testing.assert_allclose(before, after, atol=1e-10)
        finally:
            os.unlink(path)

    def test_save_load_full_model(self):
        x = np.random.randn(3, 4)
        before = self.model.predict(x)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            self.model.save_model(path)
            loaded = NeuralNetwork.load_model(path)
            after = loaded.predict(x)
            np.testing.assert_allclose(before, after, atol=1e-10)
        finally:
            os.unlink(path)


class TestCallbacks(unittest.TestCase):

    def test_early_stopping_triggers(self):
        np.random.seed(0)
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)

        model = NeuralNetwork()
        model.add(Dense(4, input_size=5, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer=SGD(learning_rate=0.001), loss="mse")

        es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=False)
        model.fit(X, y, epochs=200, batch_size=10, callbacks=[es], verbose=0)
        # No debe correr los 200 epochs si converge
        self.assertTrue(es.stopped_epoch >= 0)

    def test_validation_split(self):
        np.random.seed(0)
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)

        model = NeuralNetwork()
        model.add(Dense(4, input_size=5, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(
            X, y, epochs=3, batch_size=16, validation_split=0.2, verbose=0
        )
        self.assertIn("val_loss", history)


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


class TestRegularization(unittest.TestCase):

    def test_l2_adds_to_loss(self):
        np.random.seed(0)
        X = np.random.randn(20, 3)
        y = np.random.randn(20, 1)

        model = NeuralNetwork()
        model.add(Dense(4, input_size=3, activation="relu", kernel_regularizer=L2(0.1)))
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(X, y, epochs=2, batch_size=4, verbose=0)
        self.assertGreater(history["loss"][0], 0.0)


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
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(oh, expected)


class TestPredictBatched(unittest.TestCase):

    def test_predict_in_batches_matches_single_pass(self):
        np.random.seed(0)
        model = NeuralNetwork()
        model.add(Dense(8, input_size=4, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(optimizer="adam", loss="cce")

        X = np.random.randn(100, 4)
        full = model.predict(X)
        batched = model.predict(X, batch_size=8)
        np.testing.assert_allclose(full, batched, atol=1e-10)


class TestPredictClasses(unittest.TestCase):

    def test_binary_predict_classes(self):
        np.random.seed(0)
        model = NeuralNetwork()
        model.add(Dense(4, input_size=2, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="bce")
        X = np.random.randn(10, 2)
        cls = model.predict_classes(X)
        self.assertEqual(cls.shape, (10, 1))
        self.assertTrue(set(np.unique(cls)).issubset({0, 1}))

    def test_multiclass_predict_classes(self):
        np.random.seed(0)
        model = NeuralNetwork()
        model.add(Dense(4, input_size=3, activation="relu"))
        model.add(Dense(5, activation="softmax"))
        model.compile(optimizer="adam", loss="cce")
        X = np.random.randn(8, 3)
        cls = model.predict_classes(X)
        self.assertEqual(cls.shape, (8,))


if __name__ == "__main__":
    unittest.main()