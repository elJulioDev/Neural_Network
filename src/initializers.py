"""Inicializadores con soporte para serialización JSON."""
from typing import Dict, Any, Tuple, Optional
import numpy as np


class Initializer:
    def __call__(self, shape: Tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        return {"class_name": type(self).__name__, "config": {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Initializer":
        return cls(**config)


class HeNormal(Initializer):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape):
        rng = np.random.default_rng(self.seed)
        fan_in = shape[0]
        return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)

    def get_config(self):
        return {"class_name": "HeNormal", "config": {"seed": self.seed}}


class XavierNormal(Initializer):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape):
        rng = np.random.default_rng(self.seed)
        fan_in, fan_out = shape
        return rng.standard_normal(shape) * np.sqrt(2.0 / (fan_in + fan_out))

    def get_config(self):
        return {"class_name": "XavierNormal", "config": {"seed": self.seed}}


class XavierUniform(Initializer):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape):
        rng = np.random.default_rng(self.seed)
        fan_in, fan_out = shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape)

    def get_config(self):
        return {"class_name": "XavierUniform", "config": {"seed": self.seed}}


class Zeros(Initializer):
    def __call__(self, shape):
        return np.zeros(shape)


class Ones(Initializer):
    def __call__(self, shape):
        return np.ones(shape)


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

_INIT_CLASSES = {
    "HeNormal": HeNormal,
    "XavierNormal": XavierNormal,
    "XavierUniform": XavierUniform,
    "Zeros": Zeros,
    "Ones": Ones,
}


def get_initializer(initializer) -> Initializer:
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
    if isinstance(initializer, dict):
        cls = _INIT_CLASSES[initializer["class_name"]]
        return cls.from_config(initializer.get("config", {}))
    raise TypeError(f"Tipo no soportado: {type(initializer)}")