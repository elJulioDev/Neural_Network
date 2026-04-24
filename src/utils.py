"""
Utilidades comunes para preparación de datos.

train_test_split: separa datos en entrenamiento y validación.
to_categorical: convierte índices de clase a one-hot.
normalize: escala min-max a [0, 1].
standardize: z-score (media 0, desviación 1).
shuffle: barajado conjunto de X e y.
batch_iterator: generador de mini-batches.
"""
from typing import Tuple, Iterator
import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (X_train, X_test, y_train, y_test)."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size debe estar en (0, 1).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y deben tener el mismo número de muestras.")

    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def to_categorical(y: np.ndarray, num_classes: int = None) -> np.ndarray:
    """Convierte vector de índices a matriz one-hot."""
    y = np.asarray(y).flatten().astype(int)
    if num_classes is None:
        num_classes = int(y.max()) + 1
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def normalize(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """Escala a [0, 1] por columna."""
    X = X.astype(float)
    mn = X.min(axis=axis, keepdims=True)
    mx = X.max(axis=axis, keepdims=True)
    return (X - mn) / (mx - mn + 1e-12)


def standardize(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score por columna."""
    X = X.astype(float)
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    return (X - mean) / (std + 1e-12)


def shuffle_arrays(X: np.ndarray, y: np.ndarray, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Baraja X e y manteniendo correspondencia."""
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def batch_iterator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Genera mini-batches sobre los datos."""
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]