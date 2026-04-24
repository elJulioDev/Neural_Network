"""
Funciones de activación — versión stateless (v0.4+).

Las activaciones ya NO guardan estado en la instancia. El método
`forward` devuelve `(output, cache)` y `backward(d_output, cache)`
usa ese cache explícito. Esto permite:
  - Reutilizar una misma instancia en dos entradas (redes siamesas).
  - Threads/concurrencia sobre la misma capa.
  - Gradient check determinista.

Softmax ahora implementa el Jacobiano-vector completo, por lo que es
matemáticamente correcto con CUALQUIER función de pérdida. El atajo
(pred - y) sigue disponible vía `from_logits=True` en las losses.

Activaciones: Sigmoid, ReLU, LeakyReLU, ELU, Tanh, Softmax, Linear.
"""
from typing import Dict, Tuple, Any
import numpy as np


Cache = Dict[str, Any]


class Activation:
    """Base class stateless."""

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Cache]:
        raise NotImplementedError

    def backward(self, d_output: np.ndarray, cache: Cache) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """Config serializable a JSON. Reconstruible con from_config()."""
        return {"class_name": type(self).__name__, "config": {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Activation":
        return cls(**config)


class Sigmoid(Activation):
    def forward(self, x):
        out = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )
        return out, {"output": out}

    def backward(self, d_output, cache):
        s = cache["output"]
        return d_output * s * (1.0 - s)


class ReLU(Activation):
    def forward(self, x):
        out = np.maximum(0, x)
        return out, {"mask": (x > 0)}

    def backward(self, d_output, cache):
        return d_output * cache["mask"].astype(d_output.dtype)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x):
        out = np.where(x > 0, x, x * self.alpha)
        return out, {"positive_mask": x > 0}

    def backward(self, d_output, cache):
        grad = np.where(cache["positive_mask"], 1.0, self.alpha)
        return d_output * grad

    def get_config(self):
        return {"class_name": "LeakyReLU", "config": {"alpha": self.alpha}}


class ELU(Activation):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, x):
        out = np.where(x > 0, x, self.alpha * (np.exp(np.minimum(x, 0)) - 1))
        return out, {"x": x, "output": out}

    def backward(self, d_output, cache):
        x, out = cache["x"], cache["output"]
        grad = np.where(x > 0, 1.0, out + self.alpha)
        return d_output * grad

    def get_config(self):
        return {"class_name": "ELU", "config": {"alpha": self.alpha}}


class Tanh(Activation):
    def forward(self, x):
        out = np.tanh(x)
        return out, {"output": out}

    def backward(self, d_output, cache):
        t = cache["output"]
        return d_output * (1.0 - t ** 2)


class Softmax(Activation):
    """
    Softmax con Jacobiano-vector completo. Matemáticamente correcto con
    cualquier pérdida. Para la combinación Softmax + CategoricalCrossEntropy,
    usa `from_logits=True` en la loss para el atajo numéricamente estable.
    """

    def forward(self, x):
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        out = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return out, {"output": out}

    def backward(self, d_output, cache):
        s = cache["output"]
        dot = np.sum(d_output * s, axis=1, keepdims=True)
        return s * (d_output - dot)


class Linear(Activation):
    def forward(self, x):
        return x, {}

    def backward(self, d_output, cache):
        return d_output


_ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "elu": ELU,
    "tanh": Tanh,
    "softmax": Softmax,
    "linear": Linear,
}

_ACTIVATION_CLASSES = {
    "Sigmoid": Sigmoid,
    "ReLU": ReLU,
    "LeakyReLU": LeakyReLU,
    "ELU": ELU,
    "Tanh": Tanh,
    "Softmax": Softmax,
    "Linear": Linear,
}


def get_activation(activation) -> Activation:
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
    if isinstance(activation, dict):
        name = activation["class_name"]
        if name not in _ACTIVATION_CLASSES:
            raise ValueError(f"Activación desconocida en config: {name}")
        return _ACTIVATION_CLASSES[name].from_config(activation.get("config", {}))
    raise TypeError(f"Tipo no soportado: {type(activation)}")