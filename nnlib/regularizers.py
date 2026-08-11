"""L1, L2, L1L2 regularizers with serialization support."""
from typing import Any, Dict, Optional, Union

import numpy as np


class Regularizer:
    """Base class for weight regularizers."""

    def loss(self, weights: np.ndarray) -> float:
        """Returns the regularization penalty."""
        raise NotImplementedError

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Returns the gradient of the regularization penalty."""
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """Returns a JSON-serializable config dict."""
        return {"class_name": type(self).__name__, "config": {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Regularizer":
        """Reconstructs a regularizer from a config dict."""
        return cls(**config)


class L1(Regularizer):
    """L1 regularization: lambda * sum(|w|)."""

    def __init__(self, l1: float = 0.01):
        self.l1 = l1

    def loss(self, weights: np.ndarray) -> float:
        """Returns L1 penalty."""
        return self.l1 * np.sum(np.abs(weights))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Returns L1 gradient: lambda * sign(w)."""
        return self.l1 * np.sign(weights)

    def get_config(self) -> Dict[str, Any]:
        """Returns L1 config."""
        return {"class_name": "L1", "config": {"l1": self.l1}}


class L2(Regularizer):
    """L2 regularization: 0.5 * lambda * sum(w^2)."""

    def __init__(self, l2: float = 0.01):
        self.l2 = l2

    def loss(self, weights: np.ndarray) -> float:
        """Returns L2 penalty."""
        return 0.5 * self.l2 * np.sum(weights ** 2)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Returns L2 gradient: lambda * w."""
        return self.l2 * weights

    def get_config(self) -> Dict[str, Any]:
        """Returns L2 config."""
        return {"class_name": "L2", "config": {"l2": self.l2}}


class L1L2(Regularizer):
    """Combined L1 + L2 regularization."""

    def __init__(self, l1: float = 0.01, l2: float = 0.01):
        self.l1 = l1
        self.l2 = l2

    def loss(self, weights: np.ndarray) -> float:
        """Returns L1 + L2 penalty."""
        return self.l1 * np.sum(np.abs(weights)) + 0.5 * self.l2 * np.sum(weights ** 2)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Returns L1 + L2 gradient."""
        return self.l1 * np.sign(weights) + self.l2 * weights

    def get_config(self) -> Dict[str, Any]:
        """Returns L1L2 config."""
        return {"class_name": "L1L2", "config": {"l1": self.l1, "l2": self.l2}}


_REG_CLASSES = {"L1": L1, "L2": L2, "L1L2": L1L2}


def get_regularizer(reg: Union[str, Dict[str, Any], Regularizer, None]) -> Optional[Regularizer]:
    """Resolves a regularizer from dict, instance, or None."""
    if reg is None or isinstance(reg, Regularizer):
        return reg
    if isinstance(reg, dict):
        cls = _REG_CLASSES[reg["class_name"]]
        return cls.from_config(reg.get("config", {}))
    raise TypeError(f"Invalid regularizer: {type(reg)}")
