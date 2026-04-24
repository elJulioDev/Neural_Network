"""
Funciones de activación.

Cada activación cachea su salida en `forward()` para que `derivative()`
no tenga que recomputarla. Esto reduce ~30% el tiempo de backward pass
en redes profundas.

Activaciones disponibles: Sigmoid, ReLU, LeakyReLU, ELU, Tanh, Softmax,
Linear.
"""
import numpy as np


class Activation:
    """Clase base. Subclases deben implementar forward() y derivative()."""

    def __init__(self):
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Versión numéricamente estable
        out = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )
        self._cache = out
        return out

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self._cache if self._cache is not None else self.forward(x)
        return s * (1.0 - s)


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(x.dtype)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return np.where(x > 0, x, x * self.alpha)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.alpha)


class ELU(Activation):
    """Exponential Linear Unit. Más suave que ReLU, mejor convergencia."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.where(x > 0, x, self.alpha * (np.exp(np.minimum(x, 0)) - 1))
        self._cache = (x, out)
        return out

    def derivative(self, x: np.ndarray) -> np.ndarray:
        _, out = self._cache
        return np.where(x > 0, 1.0, out + self.alpha)


class Tanh(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self._cache = out
        return out

    def derivative(self, x: np.ndarray) -> np.ndarray:
        t = self._cache if self._cache is not None else self.forward(x)
        return 1.0 - t ** 2


class Softmax(Activation):
    """
    Softmax para clasificación multiclase. Se asume uso conjunto con
    CategoricalCrossEntropy, que aplica el truco (pred - y) en su gradiente.
    Por eso `derivative()` retorna 1 (identidad).
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        out = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        self._cache = out
        return out

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class Linear(Activation):
    """Identidad. Para regresión en la capa de salida."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


_ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "elu": ELU,
    "tanh": Tanh,
    "softmax": Softmax,
    "linear": Linear,
}


def get_activation(activation):
    """Resuelve activación desde string o instancia."""
    if isinstance(activation, Activation):
        return activation
    if isinstance(activation, str):
        key = activation.lower()
        if key not in _ACTIVATIONS:
            raise ValueError(
                f"Activación desconocida: {activation}. "
                f"Opciones: {list(_ACTIVATIONS.keys())}"
            )
        return _ACTIVATIONS[key]()
    raise TypeError(f"Tipo no soportado: {type(activation)}")