"""
Regularizadores: penalizan pesos grandes para combatir overfitting.

L1: penaliza la suma de valores absolutos (induce sparsity).
L2: penaliza la suma de cuadrados (weight decay clásico).
L1L2: combinación de ambos (Elastic Net).
"""
import numpy as np


class Regularizer:
    """Clase base. Cada regularizador define una pérdida y un gradiente."""

    def loss(self, weights: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class L1(Regularizer):
    def __init__(self, l1: float = 0.01):
        self.l1 = l1

    def loss(self, weights: np.ndarray) -> float:
        return self.l1 * np.sum(np.abs(weights))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.l1 * np.sign(weights)


class L2(Regularizer):
    def __init__(self, l2: float = 0.01):
        self.l2 = l2

    def loss(self, weights: np.ndarray) -> float:
        return 0.5 * self.l2 * np.sum(weights ** 2)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.l2 * weights


class L1L2(Regularizer):
    def __init__(self, l1: float = 0.01, l2: float = 0.01):
        self.l1 = l1
        self.l2 = l2

    def loss(self, weights: np.ndarray) -> float:
        return self.l1 * np.sum(np.abs(weights)) + 0.5 * self.l2 * np.sum(weights ** 2)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.l1 * np.sign(weights) + self.l2 * weights


def get_regularizer(reg):
    """Resuelve regularizador (None permitido)."""
    if reg is None or isinstance(reg, Regularizer):
        return reg
    raise TypeError(f"Regularizador inválido: {type(reg)}")