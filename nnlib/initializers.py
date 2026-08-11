"""Initializers with JSON serialization support."""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class Initializer:
    """Base class for weight initializers."""

    def __call__(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Returns an array of the given shape with initialized values."""
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """Returns a JSON-serializable config dict."""
        return {"class_name": type(self).__name__, "config": {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Initializer":
        """Reconstructs an initializer from a config dict."""
        return cls(**config)


class HeNormal(Initializer):
    """He Normal: N(0, sqrt(2/fan_in)). Good for ReLU layers."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generates He Normal samples."""
        rng = np.random.default_rng(self.seed)
        fan_in = shape[0]
        return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)

    def get_config(self) -> Dict[str, Any]:
        """Returns HeNormal config."""
        return {"class_name": "HeNormal", "config": {"seed": self.seed}}


class XavierNormal(Initializer):
    """Xavier Normal: N(0, sqrt(2/(fan_in + fan_out))). Good for tanh/sigmoid."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generates Xavier Normal samples."""
        rng = np.random.default_rng(self.seed)
        fan_in, fan_out = shape
        return rng.standard_normal(shape) * np.sqrt(2.0 / (fan_in + fan_out))

    def get_config(self) -> Dict[str, Any]:
        """Returns XavierNormal config."""
        return {"class_name": "XavierNormal", "config": {"seed": self.seed}}


class XavierUniform(Initializer):
    """Xavier Uniform: U(-limit, limit) where limit = sqrt(6/(fan_in + fan_out))."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generates Xavier Uniform samples."""
        rng = np.random.default_rng(self.seed)
        fan_in, fan_out = shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape)

    def get_config(self) -> Dict[str, Any]:
        """Returns XavierUniform config."""
        return {"class_name": "XavierUniform", "config": {"seed": self.seed}}


class Zeros(Initializer):
    """Returns an array of zeros."""

    def __call__(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Returns zeros."""
        return np.zeros(shape)


class Ones(Initializer):
    """Returns an array of ones."""

    def __call__(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Returns ones."""
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


def get_initializer(initializer: Union[str, Dict[str, Any], Initializer]) -> Initializer:
    """Resolves an initializer from string, dict, or instance."""
    if isinstance(initializer, Initializer):
        return initializer
    if isinstance(initializer, str):
        key = initializer.lower()
        if key not in _INITIALIZERS:
            raise ValueError(
                f"Unknown initializer: {initializer}. "
                f"Options: {list(_INITIALIZERS.keys())}"
            )
        return _INITIALIZERS[key]()
    if isinstance(initializer, dict):
        cls = _INIT_CLASSES[initializer["class_name"]]
        return cls.from_config(initializer.get("config", {}))
    raise TypeError(f"Unsupported type: {type(initializer)}")
