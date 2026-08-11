"""
Loss functions with `from_logits` support.

`from_logits=True`: the loss receives raw logits (without final activation)
and applies sigmoid or softmax internally in a numerically stable way.
Its derivative uses the mathematical shortcut (pred - y) without needing
to go through the activation derivative.

`from_logits=False` (default): the loss receives already activated
probabilities. The gradient is computed over those probabilities and the
network must propagate it back through the activation (which now does so
correctly thanks to the full Softmax Jacobian).

The flag eliminates the HIDDEN MATHEMATICAL COUPLING that assumed the
previous layer was always Softmax: now the user explicitly declares what
the loss expects to receive.

Available: MSE, MAE, Huber, BinaryCrossEntropy,
CategoricalCrossEntropy, SparseCategoricalCrossEntropy.
"""
from typing import Any, Dict

import numpy as np

_EPSILON = 1e-15


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _softmax_stable(x: np.ndarray) -> np.ndarray:
    exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def _log_softmax_stable(x: np.ndarray) -> np.ndarray:
    """log(softmax(x)) stable: x - max - log(sum(exp(x - max)))."""
    x_shift = x - np.max(x, axis=1, keepdims=True)
    return x_shift - np.log(np.sum(np.exp(x_shift), axis=1, keepdims=True))


class Loss:
    def calculate(self, output: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def derivative(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        return {"class_name": type(self).__name__, "config": {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Loss":
        return cls(**config)


class MSE(Loss):
    def calculate(self, output, y):
        return float(np.mean((y - output) ** 2))

    def derivative(self, output, y):
        return 2.0 * (output - y) / y.size


class MAE(Loss):
    def calculate(self, output, y):
        return float(np.mean(np.abs(y - output)))

    def derivative(self, output, y):
        return np.sign(output - y) / y.size


class Huber(Loss):
    """Quadratic near 0, linear far away. Robust to outliers."""

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def calculate(self, output, y):
        error = output - y
        abs_err = np.abs(error)
        quad = np.minimum(abs_err, self.delta)
        lin = abs_err - quad
        return float(np.mean(0.5 * quad ** 2 + self.delta * lin))

    def derivative(self, output, y):
        error = output - y
        abs_err = np.abs(error)
        grad = np.where(abs_err <= self.delta, error, self.delta * np.sign(error))
        return grad / y.size

    def get_config(self):
        return {"class_name": "Huber", "config": {"delta": self.delta}}


class BinaryCrossEntropy(Loss):
    """
    Binary classification.

    Args:
        from_logits: if True, expects raw logits and applies sigmoid
            internally. Gradient = (sigmoid(x) - y) / N.
            If False, expects probabilities in [0, 1].
    """

    def __init__(self, from_logits: bool = False):
        self.from_logits = from_logits

    def calculate(self, output, y):
        if self.from_logits:
            # log(1 + exp(-|x|)) + max(x, 0) - x*y — stable form
            x = output
            return float(
                np.mean(np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x))))
            )
        p = np.clip(output, _EPSILON, 1 - _EPSILON)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def derivative(self, output, y):
        if self.from_logits:
            return (_sigmoid_stable(output) - y) / y.size
        p = np.clip(output, _EPSILON, 1 - _EPSILON)
        return (p - y) / (p * (1 - p)) / y.size

    def get_config(self):
        return {"class_name": "BinaryCrossEntropy", "config": {"from_logits": self.from_logits}}


class CategoricalCrossEntropy(Loss):
    """
    Multi-class with one-hot labels.

    Args:
        from_logits: if True, expects raw logits and applies softmax
            internally. Gradient = (softmax(x) - y) / N. This is the
            numerically stable and computationally efficient path.
            If False, expects probabilities.
    """

    def __init__(self, from_logits: bool = False):
        self.from_logits = from_logits

    def calculate(self, output, y):
        if self.from_logits:
            return float(-np.sum(y * _log_softmax_stable(output)) / y.shape[0])
        p = np.clip(output, _EPSILON, 1 - _EPSILON)
        return float(-np.sum(y * np.log(p)) / y.shape[0])

    def derivative(self, output, y):
        if self.from_logits:
            return (_softmax_stable(output) - y) / y.shape[0]
        # "Honest" derivative of -sum(y*log(p)) w.r.t. p, without
        # assuming Softmax behind it: -y/p / N. The network will propagate
        # through the real Jacobian of the activation.
        p = np.clip(output, _EPSILON, 1 - _EPSILON)
        return -(y / p) / y.shape[0]

    def get_config(self):
        return {"class_name": "CategoricalCrossEntropy", "config": {"from_logits": self.from_logits}}


class SparseCategoricalCrossEntropy(Loss):
    """Multi-class with integer index labels."""

    def __init__(self, from_logits: bool = False):
        self.from_logits = from_logits

    def calculate(self, output, y):
        m = y.shape[0]
        y_idx = y.flatten().astype(int)
        if self.from_logits:
            log_probs = _log_softmax_stable(output)
            return float(-np.mean(log_probs[np.arange(m), y_idx]))
        p = np.clip(output, _EPSILON, 1 - _EPSILON)
        return float(-np.mean(np.log(p[np.arange(m), y_idx])))

    def derivative(self, output, y):
        m = y.shape[0]
        y_idx = y.flatten().astype(int)
        if self.from_logits:
            probs = _softmax_stable(output)
            probs[np.arange(m), y_idx] -= 1.0
            return probs / m
        p = np.clip(output, _EPSILON, 1 - _EPSILON)
        grad = np.zeros_like(p)
        grad[np.arange(m), y_idx] = -1.0 / p[np.arange(m), y_idx]
        return grad / m

    def get_config(self):
        return {"class_name": "SparseCategoricalCrossEntropy", "config": {"from_logits": self.from_logits}}


_LOSSES = {
    "mse": MSE,
    "mae": MAE,
    "huber": Huber,
    "binary_crossentropy": BinaryCrossEntropy,
    "bce": BinaryCrossEntropy,
    "categorical_crossentropy": CategoricalCrossEntropy,
    "cce": CategoricalCrossEntropy,
    "sparse_categorical_crossentropy": SparseCategoricalCrossEntropy,
    "scce": SparseCategoricalCrossEntropy,
}


_LOSS_CLASSES = {
    "MSE": MSE,
    "MAE": MAE,
    "Huber": Huber,
    "BinaryCrossEntropy": BinaryCrossEntropy,
    "CategoricalCrossEntropy": CategoricalCrossEntropy,
    "SparseCategoricalCrossEntropy": SparseCategoricalCrossEntropy,
}


def get_loss(loss) -> Loss:
    if isinstance(loss, Loss):
        return loss
    if isinstance(loss, str):
        key = loss.lower()
        if key not in _LOSSES:
            raise ValueError(f"Unknown loss: {loss}. Options: {list(_LOSSES.keys())}")
        return _LOSSES[key]()
    if isinstance(loss, dict):
        name = loss["class_name"]
        if name not in _LOSS_CLASSES:
            raise ValueError(f"Unknown loss in config: {name}")
        return _LOSS_CLASSES[name].from_config(loss.get("config", {}))
    raise TypeError(f"Unsupported type: {type(loss)}")
