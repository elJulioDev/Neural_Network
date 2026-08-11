"""
Multiclass demo with BatchNorm, Dropout, callbacks and from_logits.

- Final Linear layer + CategoricalCrossEntropy(from_logits=True):
  numerically stable path that avoids Softmax/Loss coupling.
- L2 regularization, gradient clipping, EarlyStopping, ReduceLROnPlateau.
- save(dir) + load(dir) with topology JSON + NPZ weights.
"""
import os
import tempfile
import numpy as np

from nnlib import (
    NeuralNetwork, Dense, Dropout, BatchNormalization,
    Adam, L2, CategoricalCrossEntropy,
    EarlyStopping, ReduceLROnPlateau,
    train_test_split, to_categorical, standardize,
)


def make_blobs(n_per_class=300, n_features=8, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, size=(n_classes, n_features))
    X, y = [], []
    for cls in range(n_classes):
        X.append(rng.normal(centers[cls], 1.0, size=(n_per_class, n_features)))
        y.extend([cls] * n_per_class)
    return np.vstack(X), np.array(y)


def main():
    np.random.seed(0)

    X, y = make_blobs(n_per_class=300, n_features=8, n_classes=4, seed=0)
    X = standardize(X)
    y_oh = to_categorical(y, num_classes=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y_oh, test_size=0.2, random_state=0)

    # Final 'linear' layer — the loss applies softmax internally
    model = NeuralNetwork()
    model.add(Dense(32, input_size=8, activation="relu", kernel_regularizer=L2(0.001)))
    model.add(BatchNormalization(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu", kernel_regularizer=L2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation="linear"))  # logits

    model.compile(
        optimizer=Adam(learning_rate=0.01, clip_norm=1.0),
        loss=CategoricalCrossEntropy(from_logits=True),
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    model.fit(
        X_train, y_train,
        epochs=100, batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluation: we need accuracy, so we apply softmax manually
    logits_test = model.predict(X_test)
    probs = np.exp(logits_test - logits_test.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    acc = np.mean(np.argmax(probs, axis=1) == np.argmax(y_test, axis=1))
    print(f"\n--- Test accuracy: {acc:.4f}")

    # Portable persistence
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "multiclass_model")
        model.save(path)
        print(f"Model saved to {path}/")
        loaded = NeuralNetwork.load(path)
        loaded_logits = loaded.predict(X_test)
        assert np.allclose(logits_test, loaded_logits)
        print("save/load JSON+NPZ: OK")


if __name__ == "__main__":
    main()