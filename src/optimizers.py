"""
Optimizadores: actualizan los pesos a partir de los gradientes.

Disponibles:
- SGD: descenso de gradiente con momentum opcional y Nesterov.
- AdaGrad: ajusta lr por parámetro acumulando gradientes al cuadrado.
- RMSprop: variante con media móvil exponencial. Bueno para RNNs.
- Adam: el caballo de batalla. Combina momentum y RMSprop. Recomendado
  por defecto para la mayoría de problemas.

Todos soportan gradient clipping (clip_norm o clip_value) para prevenir
gradientes que explotan en redes profundas.
"""
from typing import List, Optional
import numpy as np


class Optimizer:
    """
    Clase base. Subclases implementan `_update_param`.
    Maneja gradient clipping y la iteración común sobre capas entrenables.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        clip_norm: Optional[float] = None,
        clip_value: Optional[float] = None,
    ):
        self.lr = learning_rate
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        self.iterations = 0

    def _clip_gradient(self, grad: np.ndarray) -> np.ndarray:
        if self.clip_value is not None:
            grad = np.clip(grad, -self.clip_value, self.clip_value)
        if self.clip_norm is not None:
            norm = np.linalg.norm(grad)
            if norm > self.clip_norm:
                grad = grad * (self.clip_norm / (norm + 1e-12))
        return grad

    def update(self, layers: List) -> None:
        """Itera capas entrenables y actualiza pesos y biases."""
        self.iterations += 1
        for layer in layers:
            if not getattr(layer, "trainable", False):
                continue
            if not hasattr(layer, "weights"):
                continue
            layer_id = id(layer)
            dw = self._clip_gradient(layer.dweights)
            db = self._clip_gradient(layer.dbiases)
            self._update_param(layer_id, "w", layer, "weights", dw)
            self._update_param(layer_id, "b", layer, "biases", db)

    def _update_param(self, layer_id, key, layer, attr, grad):
        raise NotImplementedError


class SGD(Optimizer):
    """SGD con momentum opcional. Soporta Nesterov accelerated gradient."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        **kwargs,
    ):
        super().__init__(learning_rate, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        self._velocity = {}

    def _update_param(self, layer_id, key, layer, attr, grad):
        cache_key = (layer_id, key)
        if cache_key not in self._velocity:
            self._velocity[cache_key] = np.zeros_like(grad)
        v = self._velocity[cache_key]
        v_new = self.momentum * v - self.lr * grad
        self._velocity[cache_key] = v_new

        if self.nesterov:
            update = self.momentum * v_new - self.lr * grad
        else:
            update = v_new
        setattr(layer, attr, getattr(layer, attr) + update)


class AdaGrad(Optimizer):
    """Acumula gradientes al cuadrado. lr efectivo decae con el tiempo."""

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-7, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.epsilon = epsilon
        self._cache = {}

    def _update_param(self, layer_id, key, layer, attr, grad):
        cache_key = (layer_id, key)
        if cache_key not in self._cache:
            self._cache[cache_key] = np.zeros_like(grad)
        self._cache[cache_key] += grad ** 2
        update = -self.lr * grad / (np.sqrt(self._cache[cache_key]) + self.epsilon)
        setattr(layer, attr, getattr(layer, attr) + update)


class RMSprop(Optimizer):
    """Media móvil exponencial de gradientes al cuadrado."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(learning_rate, **kwargs)
        self.rho = rho
        self.epsilon = epsilon
        self._cache = {}

    def _update_param(self, layer_id, key, layer, attr, grad):
        cache_key = (layer_id, key)
        if cache_key not in self._cache:
            self._cache[cache_key] = np.zeros_like(grad)
        self._cache[cache_key] = (
            self.rho * self._cache[cache_key] + (1 - self.rho) * grad ** 2
        )
        update = -self.lr * grad / (np.sqrt(self._cache[cache_key]) + self.epsilon)
        setattr(layer, attr, getattr(layer, attr) + update)


class Adam(Optimizer):
    """
    Adaptive Moment Estimation. Mantiene primer y segundo momento de los
    gradientes con bias correction. Por defecto la mejor elección general.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(learning_rate, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self._m = {}  # primer momento
        self._v = {}  # segundo momento

    def _update_param(self, layer_id, key, layer, attr, grad):
        cache_key = (layer_id, key)
        if cache_key not in self._m:
            self._m[cache_key] = np.zeros_like(grad)
            self._v[cache_key] = np.zeros_like(grad)

        self._m[cache_key] = self.beta_1 * self._m[cache_key] + (1 - self.beta_1) * grad
        self._v[cache_key] = self.beta_2 * self._v[cache_key] + (1 - self.beta_2) * grad ** 2

        # Bias correction
        m_hat = self._m[cache_key] / (1 - self.beta_1 ** self.iterations)
        v_hat = self._v[cache_key] / (1 - self.beta_2 ** self.iterations)

        update = -self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        setattr(layer, attr, getattr(layer, attr) + update)


_OPTIMIZERS = {
    "sgd": SGD,
    "adagrad": AdaGrad,
    "rmsprop": RMSprop,
    "adam": Adam,
}


def get_optimizer(opt):
    if isinstance(opt, Optimizer):
        return opt
    if isinstance(opt, str):
        key = opt.lower()
        if key not in _OPTIMIZERS:
            raise ValueError(f"Optimizer desconocido: {opt}. Opciones: {list(_OPTIMIZERS.keys())}")
        return _OPTIMIZERS[key]()
    raise TypeError(f"Tipo no soportado: {type(opt)}")