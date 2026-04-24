"""
Optimizadores v0.4 — API genérica independiente del tipo de parámetro.

Elimina las fugas de abstracción de v0.3: los optimizadores ya no
inspeccionan atributos hardcodeados (`weights`, `biases`, etc). Reciben
tuplas `(layer_id, param_name, param_array, grad_array)` y aplican la
regla de actualización in-place.

Esto permite capas con cualquier número y nombre de parámetros sin
tocar el código del optimizer:
  - Dense: {'weights', 'biases'}
  - BatchNormalization: {'gamma', 'beta'}
  - CualquierCapaFutura: {'query', 'key', 'value', 'out_proj', ...}

Todos soportan gradient clipping (clip_norm, clip_value).
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Optimizer:
    """
    Base.

    Subclases implementan `_update(layer_id, name, param, grad)`, que
    debe modificar `param` IN-PLACE (usando `param += delta` o
    equivalente).
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

    def _clip(self, grad: np.ndarray) -> np.ndarray:
        if self.clip_value is not None:
            grad = np.clip(grad, -self.clip_value, self.clip_value)
        if self.clip_norm is not None:
            norm = np.linalg.norm(grad)
            if norm > self.clip_norm:
                grad = grad * (self.clip_norm / (norm + 1e-12))
        return grad

    def apply_gradients(
        self,
        grads_and_params: List[Tuple[int, str, np.ndarray, np.ndarray]],
    ) -> None:
        """
        Aplica gradientes a sus parámetros.

        Args:
            grads_and_params: lista de tuplas
                (layer_id, param_name, param_array, grad_array).
                `param_array` debe ser la referencia al array real de
                la capa para que la actualización in-place sea efectiva.
        """
        self.iterations += 1
        for layer_id, name, param, grad in grads_and_params:
            g = self._clip(grad)
            self._update(layer_id, name, param, g)

    def _update(self, layer_id: int, name: str, param: np.ndarray, grad: np.ndarray) -> None:
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        return {
            "class_name": type(self).__name__,
            "config": {
                "learning_rate": self.lr,
                "clip_norm": self.clip_norm,
                "clip_value": self.clip_value,
            },
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Optimizer":
        return cls(**config)


class SGD(Optimizer):
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
        self._velocity: Dict[Tuple[int, str], np.ndarray] = {}

    def _update(self, layer_id, name, param, grad):
        key = (layer_id, name)
        if key not in self._velocity:
            self._velocity[key] = np.zeros_like(grad)
        v = self._velocity[key]
        v_new = self.momentum * v - self.lr * grad
        self._velocity[key] = v_new

        if self.nesterov:
            update = self.momentum * v_new - self.lr * grad
        else:
            update = v_new
        param += update  # in-place

    def get_config(self):
        cfg = super().get_config()
        cfg["config"].update({"momentum": self.momentum, "nesterov": self.nesterov})
        return cfg


class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-7, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.epsilon = epsilon
        self._cache: Dict[Tuple[int, str], np.ndarray] = {}

    def _update(self, layer_id, name, param, grad):
        key = (layer_id, name)
        if key not in self._cache:
            self._cache[key] = np.zeros_like(grad)
        self._cache[key] += grad ** 2
        param -= self.lr * grad / (np.sqrt(self._cache[key]) + self.epsilon)

    def get_config(self):
        cfg = super().get_config()
        cfg["config"]["epsilon"] = self.epsilon
        return cfg


class RMSprop(Optimizer):
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
        self._cache: Dict[Tuple[int, str], np.ndarray] = {}

    def _update(self, layer_id, name, param, grad):
        key = (layer_id, name)
        if key not in self._cache:
            self._cache[key] = np.zeros_like(grad)
        self._cache[key] = self.rho * self._cache[key] + (1 - self.rho) * grad ** 2
        param -= self.lr * grad / (np.sqrt(self._cache[key]) + self.epsilon)

    def get_config(self):
        cfg = super().get_config()
        cfg["config"].update({"rho": self.rho, "epsilon": self.epsilon})
        return cfg


class Adam(Optimizer):
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
        self._m: Dict[Tuple[int, str], np.ndarray] = {}
        self._v: Dict[Tuple[int, str], np.ndarray] = {}

    def _update(self, layer_id, name, param, grad):
        key = (layer_id, name)
        if key not in self._m:
            self._m[key] = np.zeros_like(grad)
            self._v[key] = np.zeros_like(grad)

        self._m[key] = self.beta_1 * self._m[key] + (1 - self.beta_1) * grad
        self._v[key] = self.beta_2 * self._v[key] + (1 - self.beta_2) * grad ** 2

        m_hat = self._m[key] / (1 - self.beta_1 ** self.iterations)
        v_hat = self._v[key] / (1 - self.beta_2 ** self.iterations)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def get_config(self):
        cfg = super().get_config()
        cfg["config"].update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
        })
        return cfg


_OPTIMIZERS = {"sgd": SGD, "adagrad": AdaGrad, "rmsprop": RMSprop, "adam": Adam}
_OPT_CLASSES = {"SGD": SGD, "AdaGrad": AdaGrad, "RMSprop": RMSprop, "Adam": Adam}


def get_optimizer(opt) -> Optimizer:
    if isinstance(opt, Optimizer):
        return opt
    if isinstance(opt, str):
        key = opt.lower()
        if key not in _OPTIMIZERS:
            raise ValueError(f"Optimizer desconocido: {opt}. Opciones: {list(_OPTIMIZERS.keys())}")
        return _OPTIMIZERS[key]()
    if isinstance(opt, dict):
        cls = _OPT_CLASSES[opt["class_name"]]
        return cls.from_config(opt.get("config", {}))
    raise TypeError(f"Tipo no soportado: {type(opt)}")