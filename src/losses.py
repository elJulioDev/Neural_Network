"""
Funciones de pérdida con soporte `from_logits`.

`from_logits=True`: la pérdida recibe los logits crudos (sin activación
final) y aplica internamente sigmoid o softmax de forma numéricamente
estable. Su derivada usa el atajo matemático (pred - y) sin necesidad de
pasar por la derivada de la activación.

`from_logits=False` (default): la pérdida recibe probabilidades ya
activadas. El gradiente se calcula sobre esas probabilidades y la red
debe propagarlo hacia atrás por la activación (que ahora lo hace
correctamente gracias al Jacobiano completo de Softmax).

El flag elimina el ACOPLAMIENTO MATEMÁTICO OCULTO que asumía que la
capa anterior siempre era Softmax: ahora el usuario declara
explícitamente qué espera recibir la loss.

Disponibles: MSE, MAE, Huber, BinaryCrossEntropy,
CategoricalCrossEntropy, SparseCategoricalCrossEntropy.
"""
from typing import Dict, Any
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
    """log(softmax(x)) estable: x - max - log(sum(exp(x - max)))."""
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
    """Cuadrático cerca de 0, lineal lejos. Robusto a outliers."""

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
    Clasificación binaria.

    Args:
        from_logits: si True, espera logits crudos y aplica sigmoid
            internamente. Gradiente = (sigmoid(x) - y) / N.
            Si False, espera probabilidades en [0, 1].
    """

    def __init__(self, from_logits: bool = False):
        self.from_logits = from_logits

    def calculate(self, output, y):
        if self.from_logits:
            # log(1 + exp(-|x|)) + max(x, 0) - x*y — forma estable
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
    Multiclase con labels one-hot.

    Args:
        from_logits: si True, espera logits crudos y aplica softmax
            internamente. Gradiente = (softmax(x) - y) / N. Este es el
            camino numéricamente estable y computacionalmente eficiente.
            Si False, espera probabilidades.
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
        # Derivada "honesta" de -sum(y*log(p)) respecto a p, sin
        # asumir Softmax atrás: -y/p / N. La red propagará por el
        # Jacobiano real de la activación.
        p = np.clip(output, _EPSILON, 1 - _EPSILON)
        return -(y / p) / y.shape[0]

    def get_config(self):
        return {"class_name": "CategoricalCrossEntropy", "config": {"from_logits": self.from_logits}}


class SparseCategoricalCrossEntropy(Loss):
    """Multiclase con labels como índices enteros."""

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
            raise ValueError(f"Loss desconocida: {loss}. Opciones: {list(_LOSSES.keys())}")
        return _LOSSES[key]()
    if isinstance(loss, dict):
        name = loss["class_name"]
        if name not in _LOSS_CLASSES:
            raise ValueError(f"Loss desconocida en config: {name}")
        return _LOSS_CLASSES[name].from_config(loss.get("config", {}))
    raise TypeError(f"Tipo no soportado: {type(loss)}")