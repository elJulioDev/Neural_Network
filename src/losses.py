"""
Funciones de pérdida.

MSE: regresión.
MAE: regresión robusta a outliers.
Huber: combina lo mejor de MSE y MAE.
BinaryCrossEntropy: clasificación binaria.
CategoricalCrossEntropy: multiclase con etiquetas one-hot.
SparseCategoricalCrossEntropy: multiclase con etiquetas como enteros.
"""
import numpy as np

_EPSILON = 1e-15


class Loss:
    def calculate(self, output: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def derivative(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MSE(Loss):
    def calculate(self, output, y):
        return float(np.mean((y - output) ** 2))

    def derivative(self, output, y):
        # np.mean divide por y.size (todos los elementos), no por batch
        return 2.0 * (output - y) / y.size


class MAE(Loss):
    def calculate(self, output, y):
        return float(np.mean(np.abs(y - output)))

    def derivative(self, output, y):
        return np.sign(output - y) / y.size


class Huber(Loss):
    """
    Pérdida de Huber: cuadrática para errores pequeños, lineal para grandes.
    Ofrece estabilidad de MSE con robustez de MAE frente a outliers.
    """

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
        # Cuadrático en zona |e| <= delta, lineal fuera
        grad = np.where(abs_err <= self.delta, error, self.delta * np.sign(error))
        return grad / y.size


class BinaryCrossEntropy(Loss):
    def calculate(self, output, y):
        output = np.clip(output, _EPSILON, 1 - _EPSILON)
        return float(-np.mean(y * np.log(output) + (1 - y) * np.log(1 - output)))

    def derivative(self, output, y):
        output = np.clip(output, _EPSILON, 1 - _EPSILON)
        return (output - y) / (output * (1 - output)) / y.shape[0]


class CategoricalCrossEntropy(Loss):
    """Etiquetas en formato one-hot. Combina con Softmax en la salida."""

    def calculate(self, output, y):
        output = np.clip(output, _EPSILON, 1 - _EPSILON)
        return float(-np.sum(y * np.log(output)) / y.shape[0])

    def derivative(self, output, y):
        # Gradiente simplificado asumiendo Softmax como activación previa
        return (output - y) / y.shape[0]


class SparseCategoricalCrossEntropy(Loss):
    """
    Como CategoricalCrossEntropy pero acepta etiquetas enteras
    (clase index) en vez de one-hot. Más eficiente con muchas clases.
    """

    def calculate(self, output, y):
        output = np.clip(output, _EPSILON, 1 - _EPSILON)
        m = y.shape[0]
        # y: shape (m,) o (m, 1) con índices de clase
        y_idx = y.flatten().astype(int)
        log_probs = -np.log(output[np.arange(m), y_idx])
        return float(np.mean(log_probs))

    def derivative(self, output, y):
        m = y.shape[0]
        y_idx = y.flatten().astype(int)
        grad = output.copy()
        grad[np.arange(m), y_idx] -= 1.0
        return grad / m


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


def get_loss(loss):
    if isinstance(loss, Loss):
        return loss
    if isinstance(loss, str):
        key = loss.lower()
        if key not in _LOSSES:
            raise ValueError(f"Loss desconocida: {loss}. Opciones: {list(_LOSSES.keys())}")
        return _LOSSES[key]()
    raise TypeError(f"Tipo no soportado: {type(loss)}")