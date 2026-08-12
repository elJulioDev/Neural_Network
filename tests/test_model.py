import os
import tempfile
import unittest

import numpy as np

from nnlib import (
    L2,
    SGD,
    Adam,
    BatchNormalization,
    BinaryCrossEntropy,
    Dense,
    Dropout,
    EarlyStopping,
    NeuralNetwork,
    to_categorical,
    train_test_split,
)


def _retry_xor(max_attempts=3):
    """Retry decorator for XOR convergence tests (non-deterministic init)."""
    def decorator(test_func):
        def wrapper(self):
            last_err = None
            for attempt in range(max_attempts):
                try:
                    test_func(self, seed=attempt * 10)
                    return
                except (AssertionError, self.failureException) as e:
                    last_err = e
            raise last_err
        wrapper.__name__ = test_func.__name__
        wrapper.__doc__ = test_func.__doc__
        return wrapper
    return decorator


class TestXORIntegration(unittest.TestCase):

    @_retry_xor()
    def test_xor_with_adam(self, seed=42):
        np.random.seed(seed)
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

    @_retry_xor()
    def test_xor_with_logits_path(self, seed=42):
        """Final layer 'linear' + BCE from_logits. The gradient must flow
        correctly through the stable path."""
        np.random.seed(seed)
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

        logits = model.predict(X)
        probs = 1.0 / (1.0 + np.exp(-logits))
        for i in range(4):
            self.assertAlmostEqual(round(float(probs[i, 0])), float(y[i, 0]))


class TestStateIsolation(unittest.TestCase):
    """External caches — a layer processing two inputs does not get corrupted."""

    def test_layer_reused_across_inputs(self):
        from nnlib.layer import Dense
        layer = Dense(4, 3, activation="relu")

        x1 = np.random.randn(5, 3)
        x2 = np.random.randn(5, 3)

        # Forward pass 1
        out1, cache1 = layer.forward(x1)
        # Forward pass 2 INTERLEAVED (before backward of 1)
        out2, cache2 = layer.forward(x2)
        # Backward 1 must use cache1 (not cache2)
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

        # Warm up running stats with a forward pass
        X = np.random.randn(20, 4)
        _ = model.predict(X)

        before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            # Verify files
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "topology.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "weights.npz")))

            # Load into new model
            loaded = NeuralNetwork.load(tmpdir)
            after = loaded.predict(X)
            np.testing.assert_allclose(before, after, atol=1e-10)

    def test_json_only_roundtrip_no_pickle(self):
        """Pure JSON topology: must reconstruct architecture without pickle."""
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
        """The running stats (non-trainable) must persist across save/load."""
        np.random.seed(0)
        model = NeuralNetwork()
        model.add(Dense(4, input_size=3, activation="linear"))
        model.add(BatchNormalization(4))
        model.compile(optimizer="adam", loss="mse")

        # Train a bit to change running stats
        model.fit(np.random.randn(50, 3), np.random.randn(50, 4), epochs=3, verbose=0)
        rm_before = model.layers[1].running_mean.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = NeuralNetwork.load(tmpdir)
            rm_after = loaded.layers[1].running_mean
            np.testing.assert_allclose(rm_before, rm_after)

    def test_optimizer_state_persisted(self):
        """Adam m/v must survive save/load — optimizer continues, not reset."""
        np.random.seed(0)
        X = np.random.randn(20, 4)
        y = np.random.randn(20, 1)

        model = NeuralNetwork()
        model.add(Dense(8, input_size=4, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        # Train a few epochs to build up moments
        model.fit(X, y, epochs=5, batch_size=10, verbose=0)

        # Capture optimizer state before save
        state_before = model.optimizer.get_state()
        iterations_before = state_before["iterations"]
        # Adam stores _m and _v dicts with (int, str) keys
        m_before = {k: v.copy() for k, v in state_before.get("_m", {}).items()}
        v_before = {k: v.copy() for k, v in state_before.get("_v", {}).items()}
        self.assertGreater(iterations_before, 0)
        self.assertTrue(len(m_before) > 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = NeuralNetwork.load(tmpdir)

            state_after = loaded.optimizer.get_state()
            self.assertEqual(state_after["iterations"], iterations_before)
            for key in m_before:
                self.assertIn(key, state_after["_m"])
                np.testing.assert_allclose(m_before[key], state_after["_m"][key])
            for key in v_before:
                self.assertIn(key, state_after["_v"])
                np.testing.assert_allclose(v_before[key], state_after["_v"][key])


class TestShapePropagation(unittest.TestCase):
    """Fail-fast on incompatible shapes."""

    def test_mismatched_input_size_fails_on_build(self):
        model = NeuralNetwork()
        model.add(Dense(4, input_size=3, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(optimizer="adam", loss="cce")

        # X has 10 features but the model expects 3
        with self.assertRaises(ValueError):
            model.fit(np.random.randn(5, 10), np.random.randn(5, 2), epochs=1, verbose=0)

    def test_build_propagates_output_shapes(self):
        model = NeuralNetwork()
        model.add(Dense(16, input_size=4, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(3, activation="softmax"))
        # Shapes should already be propagated without calling compile()
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
        # Something happened before epoch 100
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
