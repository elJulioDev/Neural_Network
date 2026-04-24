"""
Demo XOR — v0.4.

Muestra el camino recomendado de producción:
- capa final 'linear' + BinaryCrossEntropy(from_logits=True).
  Es numéricamente estable y evita el acoplamiento Softmax/Loss.
- build() propaga shapes antes de entrenar (fail-fast).
- save() genera topology.json + weights.npz sin pickle.
"""
import os
import tempfile
import numpy as np

from src import (
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

    # Inferencia: aplicar sigmoid al logit
    logits = model.predict(X)
    probs = 1.0 / (1.0 + np.exp(-logits))

    print("\n--- Predicciones ---")
    for i in range(len(X)):
        print(
            f"Input: {X[i]}, Esperado: {int(y[i, 0])}, "
            f"Prob: {float(probs[i, 0]):.4f}, "
            f"Clase: {int(probs[i, 0] >= 0.5)}"
        )

    # Persistencia portable (JSON + NPZ)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "xor_model")
        model.save(path)
        loaded = NeuralNetwork.load(path)
        assert np.allclose(model.predict(X), loaded.predict(X))
        print("\nsave/load JSON+NPZ: OK")


if __name__ == "__main__":
    main()