"""Regularizadores L1, L2, L1L2 con soporte de serialización."""
from typing import Dict, Any
import numpy as np


class Regularizer:
    def loss(self, weights: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        return {"class_name": type(self).__name__, "config": {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Regularizer":
        return cls(**config)


class L1(Regularizer):
    def __init__(self, l1: float = 0.01):
        self.l1 = l1

    def loss(self, weights):
        return self.l1 * np.sum(np.abs(weights))

    def gradient(self, weights):
        return self.l1 * np.sign(weights)

    def get_config(self):
        return {"class_name": "L1", "config": {"l1": self.l1}}


class L2(Regularizer):
    def __init__(self, l2: float = 0.01):
        self.l2 = l2

    def loss(self, weights):
        return 0.5 * self.l2 * np.sum(weights ** 2)

    def gradient(self, weights):
        return self.l2 * weights

    def get_config(self):
        return {"class_name": "L2", "config": {"l2": self.l2}}


class L1L2(Regularizer):
    def __init__(self, l1: float = 0.01, l2: float = 0.01):
        self.l1 = l1
        self.l2 = l2

    def loss(self, weights):
        return self.l1 * np.sum(np.abs(weights)) + 0.5 * self.l2 * np.sum(weights ** 2)

    def gradient(self, weights):
        return self.l1 * np.sign(weights) + self.l2 * weights

    def get_config(self):
        return {"class_name": "L1L2", "config": {"l1": self.l1, "l2": self.l2}}


_REG_CLASSES = {"L1": L1, "L2": L2, "L1L2": L1L2}


def get_regularizer(reg):
    if reg is None or isinstance(reg, Regularizer):
        return reg
    if isinstance(reg, dict):
        cls = _REG_CLASSES[reg["class_name"]]
        return cls.from_config(reg.get("config", {}))
    raise TypeError(f"Regularizador inválido: {type(reg)}")