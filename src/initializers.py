"""
Inicializadores de pesos para redes neuronales.

Cada inicializador implementa una estrategia matemática para asignar
los valores iniciales de los pesos. Una buena inicialización es CRÍTICA
para evitar problemas de gradientes que explotan o desaparecen.
"""
import numpy as np
from typing import Tuple


class Initializer:
    """Clase base para todos los inicializadores."""

    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        raise NotImplementedError


class HeNormal(Initializer):
    """
    Inicialización de He (Kaiming) para activaciones ReLU/LeakyReLU.
    Var(W) = 2 / fan_in
    """

    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        fan_in = shape[0]
        return np.random.randn(*shape) * np.sqrt(2.0 / fan_in)


class XavierNormal(Initializer):
    """
    Inicialización de Xavier/Glorot para activaciones tanh/sigmoid.
    Var(W) = 2 / (fan_in + fan_out)
    """

    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        fan_in, fan_out = shape
        return np.random.randn(*shape) * np.sqrt(2.0 / (fan_in + fan_out))


class XavierUniform(Initializer):
    """Variante uniforme de Xavier/Glorot."""

    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        fan_in, fan_out = shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)


class Zeros(Initializer):
    """Inicializa todo a cero. Usado típicamente para biases."""

    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        return np.zeros(shape)


class Ones(Initializer):
    """Inicializa todo a uno. Usado para parámetros gamma de BatchNorm."""

    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        return np.ones(shape)


# Mapping para usar strings (estilo Keras)
_INITIALIZERS = {
    "he_normal": HeNormal,
    "he": HeNormal,
    "xavier_normal": XavierNormal,
    "xavier": XavierNormal,
    "glorot_normal": XavierNormal,
    "xavier_uniform": XavierUniform,
    "glorot_uniform": XavierUniform,
    "zeros": Zeros,
    "ones": Ones,
}


def get_initializer(initializer):
    """Resuelve un inicializador desde string o instancia."""
    if isinstance(initializer, Initializer):
        return initializer
    if isinstance(initializer, str):
        key = initializer.lower()
        if key not in _INITIALIZERS:
            raise ValueError(
                f"Inicializador desconocido: {initializer}. "
                f"Opciones: {list(_INITIALIZERS.keys())}"
            )
        return _INITIALIZERS[key]()
    raise TypeError(f"Tipo no soportado: {type(initializer)}")