"""
XOR demo — v1.0.

Shows the recommended production path:
- final 'linear' layer + BinaryCrossEntropy(from_logits=True).
  It is numerically stable and avoids Softmax/Loss coupling.
- build() propagates shapes before training (fail-fast).
- save() generates topology.json + weights.npz without pickle.
"""
import os
import tempfile
import numpy as np

from nnlib import (
    NeuralNetwork, Dense,
    Adam, BinaryCrossEntropy,
    EarlyStopping,
)


def main():
    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    model = NeuralNetwork()
    model.add(Dense(8, input_size=2, activation="relu"))
    model.add(Dense(1, activation="linear"))   # logits

    model.compile(
        optimizer=Adam(learning_rate=0.05),
        loss=BinaryCrossEntropy(from_logits=True),
        metrics=[],
    )

    model.summary()

    model.fit(
        X, y,
        epochs=500,
        batch_size=4,
        callbacks=[EarlyStopping(monitor="loss", patience=50)],
        verbose=0,
    )

    # Inference: apply sigmoid to the logit
    logits = model.predict(X)
    probs = 1.0 / (1.0 + np.exp(-logits))

    print("\n--- Predictions ---")
    for i in range(len(X)):
        print(
            f"Input: {X[i]}, Expected: {int(y[i, 0])}, "
            f"Prob: {float(probs[i, 0]):.4f}, "
            f"Class: {int(probs[i, 0] >= 0.5)}"
        )

    # Portable persistence (JSON + NPZ)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "xor_model")
        model.save(path)
        loaded = NeuralNetwork.load(path)
        assert np.allclose(model.predict(X), loaded.predict(X))
        print("\nsave/load JSON+NPZ: OK")


if __name__ == "__main__":
    main()