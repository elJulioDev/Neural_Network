"""
Common utilities for data preparation.

train_test_split: splits data into training and validation sets.
to_categorical: converts class indices to one-hot.
normalize: scales min-max to [0, 1].
standardize: z-score (mean 0, std 1).
shuffle: shuffles X and y together.
batch_iterator: mini-batch generator.
"""
from typing import Iterator, Tuple

import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_train, X_test, y_train, y_test)."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def to_categorical(y: np.ndarray, num_classes: int = None) -> np.ndarray:
    """Converts index vector to one-hot matrix."""
    y = np.asarray(y).flatten().astype(int)
    if num_classes is None:
        num_classes = int(y.max()) + 1
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def normalize(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """Scales to [0, 1] per column."""
    X = X.astype(float)
    mn = X.min(axis=axis, keepdims=True)
    mx = X.max(axis=axis, keepdims=True)
    return (X - mn) / (mx - mn + 1e-12)


def standardize(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score per column."""
    X = X.astype(float)
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    return (X - mean) / (std + 1e-12)


def shuffle_arrays(X: np.ndarray, y: np.ndarray, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffles X and y while maintaining correspondence."""
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def batch_iterator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generates mini-batches over the data."""
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]
