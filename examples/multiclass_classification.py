"""
Demo avanzado: clasificación multiclase con dataset sintético.

Muestra:
- BatchNormalization y Dropout combinados.
- Optimizer Adam con gradient clipping.
- Regularización L2 sobre los pesos.
- validation_split para monitorear sobreajuste.
- EarlyStopping y ReduceLROnPlateau.
- Predicción y evaluación final.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    NeuralNetwork, Dense, Dropout, BatchNormalization,
    Adam, L2,
    EarlyStopping, ReduceLROnPlateau,
    train_test_split, to_categorical, standardize,
)


def make_blobs(n_per_class: int = 200, n_features: int = 4, n_classes: int = 3, seed: int = 0):
    """Genera clusters gaussianos en distintos centros (dataset sintético)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, size=(n_classes, n_features))
    X, y = [], []
    for cls in range(n_classes):
        X.append(rng.normal(centers[cls], 1.0, size=(n_per_class, n_features)))
        y.extend([cls] * n_per_class)
    return np.vstack(X), np.array(y)


def main():
    np.random.seed(0)

    # Datos
    X, y = make_blobs(n_per_class=300, n_features=8, n_classes=4, seed=0)
    X = standardize(X)
    y_oh = to_categorical(y, num_classes=4)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_oh, test_size=0.2, random_state=0
    )

    # Modelo profundo con regularización
    model = NeuralNetwork()
    model.add(Dense(32, input_size=8, activation="relu", kernel_regularizer=L2(0.001)))
    model.add(BatchNormalization(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu", kernel_regularizer=L2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.01, clip_norm=1.0),
        loss="cce",
        metrics=["categorical_accuracy"],
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    # Entrenar
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluar en test set
    print("\n--- Test set ---")
    model.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()