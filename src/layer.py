"""
Capas — versión v0.4 con estado externo y API genérica de parámetros.

Cambios clave respecto a v0.3:

1. `forward(inputs, training) -> (output, cache)`: ya no muta `self`. El
   cache es un dict con todo lo que `backward` necesita. Esto permite
   reutilizar una misma capa en dos entradas (redes siamesas) sin que la
   segunda pisotee el cache de la primera.

2. `backward(d_output, cache) -> (d_input, gradients_dict)`: devuelve
   los gradientes en un diccionario nombrado por parámetro. El optimizer
   ya no busca atributos hardcodeados `weights`/`biases`.

3. `parameters() -> Dict[str, ndarray]`: el optimizer recibe un
   diccionario genérico. Una capa puede tener 1, 2, 3 ó N parámetros
   distintos sin tocar el código del optimizer.

4. `build(input_shape) -> output_shape`: las capas declaran su shape de
   salida dado un shape de entrada. El modelo propaga shapes antes de
   entrenar, así las incompatibilidades dimensionales se detectan en
   `compile()` en vez de en la primera multiplicación matricial.

5. `get_config() / from_config()`: serialización JSON sin pickle.

Capas: Dense (alias Layer), Dropout, BatchNormalization.
"""
from typing import Optional, Dict, Tuple, Any
import numpy as np

from .activations import Activation, get_activation
from .initializers import get_initializer
from .regularizers import Regularizer, get_regularizer


Cache = Dict[str, Any]
ParamDict = Dict[str, np.ndarray]


class BaseLayer:
    """
    Interfaz común.

    Contrato:
    - `build(input_shape)` se llama UNA vez antes del primer forward.
      Puede llamarse con `input_shape=None` si no requiere construir
      pesos (Dropout). Retorna el shape de salida.
    - `forward` y `backward` son idealmente puros respecto al estado
      entrenable. Para estadísticas no-entrenables (running_mean en
      BatchNorm) sí se permite mutación.
    - `parameters()` devuelve un dict de REFERENCIAS (no copias) para
      que el optimizer modifique in-place.
    """

    trainable: bool = False
    # Shape de entrada/salida (fijados en build). (None, features).
    input_shape: Optional[Tuple[Optional[int], int]] = None
    output_shape: Optional[Tuple[Optional[int], int]] = None

    def build(self, input_shape: Tuple[Optional[int], int]) -> Tuple[Optional[int], int]:
        self.input_shape = input_shape
        self.output_shape = input_shape  # default: identidad
        return self.output_shape

    def forward(self, inputs: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Cache]:
        raise NotImplementedError

    def backward(self, d_output: np.ndarray, cache: Cache) -> Tuple[np.ndarray, ParamDict]:
        raise NotImplementedError

    def parameters(self) -> ParamDict:
        """Parámetros ENTRENABLES (dict de referencias)."""
        return {}

    def non_trainable_state(self) -> ParamDict:
        """Estado no entrenable persistente (ej. running_mean)."""
        return {}

    def regularization_loss(self) -> float:
        return 0.0

    def get_config(self) -> Dict[str, Any]:
        return {"class_name": type(self).__name__, "config": {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseLayer":
        return cls(**config)


class Layer(BaseLayer):
    """
    Capa totalmente conectada (Dense).

    Parámetros entrenables: 'weights', 'biases'.

    Args:
        n_neurons: tamaño de salida.
        input_size: opcional. Si se omite, se infiere en build().
        activation: instancia, string ('relu', 'sigmoid'...) o dict de config.
        kernel_initializer: estrategia de init para los pesos.
        bias_initializer: estrategia de init para los biases.
        kernel_regularizer: regularizador opcional sobre los pesos.
    """

    trainable = True

    def __init__(
        self,
        n_neurons: int,
        input_size: Optional[int] = None,
        activation="linear",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer: Optional[Regularizer] = None,
    ):
        self.n_neurons = n_neurons
        self._explicit_input_size = input_size
        self.activation: Activation = get_activation(activation)
        self._kernel_initializer = get_initializer(kernel_initializer)
        self._bias_initializer = get_initializer(bias_initializer)
        self.kernel_regularizer = get_regularizer(kernel_regularizer)

        self.weights: Optional[np.ndarray] = None
        self.biases: Optional[np.ndarray] = None
        self._built = False

        if input_size is not None:
            self.build((None, input_size))

    def build(self, input_shape):
        input_size = input_shape[-1]
        if input_size is None:
            raise ValueError(
                f"{type(self).__name__}: input_shape debe tener dimensión de features definida."
            )
        self.weights = self._kernel_initializer((input_size, self.n_neurons))
        self.biases = self._bias_initializer((1, self.n_neurons))
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], self.n_neurons)
        self._built = True
        return self.output_shape

    def forward(self, inputs, training=True):
        if not self._built:
            self.build((None, inputs.shape[1]))
        # Validación fail-fast de shape
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"{type(self).__name__}: esperaba features={self.weights.shape[0]}, "
                f"recibió {inputs.shape[1]}."
            )
        z = inputs @ self.weights + self.biases
        output, act_cache = self.activation.forward(z)
        cache = {"inputs": inputs, "z": z, "activation_cache": act_cache}
        return output, cache

    def backward(self, d_output, cache):
        # Propaga por la activación primero (usando el cache de la activación)
        d_z = self.activation.backward(d_output, cache["activation_cache"])
        inputs = cache["inputs"]

        dweights = inputs.T @ d_z
        dbiases = np.sum(d_z, axis=0, keepdims=True)

        # Regularización sobre los pesos (no sobre biases)
        if self.kernel_regularizer is not None:
            dweights = dweights + self.kernel_regularizer.gradient(self.weights)

        d_input = d_z @ self.weights.T
        grads = {"weights": dweights, "biases": dbiases}
        return d_input, grads

    def parameters(self):
        return {"weights": self.weights, "biases": self.biases}

    def regularization_loss(self):
        if self.kernel_regularizer is None:
            return 0.0
        return self.kernel_regularizer.loss(self.weights)

    def get_config(self):
        return {
            "class_name": "Dense",
            "config": {
                "n_neurons": self.n_neurons,
                "input_size": self._explicit_input_size,
                "activation": self.activation.get_config(),
                "kernel_initializer": self._kernel_initializer.get_config(),
                "bias_initializer": self._bias_initializer.get_config(),
                "kernel_regularizer": (
                    self.kernel_regularizer.get_config()
                    if self.kernel_regularizer is not None
                    else None
                ),
            },
        }


# Alias estilo Keras
Dense = Layer


class Dropout(BaseLayer):
    """
    Dropout invertido (escala 1/keep_prob en training). No tiene
    parámetros entrenables. Identidad en inferencia.
    """

    trainable = False

    def __init__(self, rate: float):
        if not 0.0 <= rate < 1.0:
            raise ValueError("rate debe estar en [0, 1).")
        self.rate = rate

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        return input_shape

    def forward(self, inputs, training=True):
        if not training or self.rate == 0.0:
            return inputs, {"mask": None}
        keep_prob = 1.0 - self.rate
        mask = (np.random.rand(*inputs.shape) < keep_prob) / keep_prob
        return inputs * mask, {"mask": mask}

    def backward(self, d_output, cache):
        mask = cache["mask"]
        if mask is None:
            return d_output, {}
        return d_output * mask, {}

    def get_config(self):
        return {"class_name": "Dropout", "config": {"rate": self.rate}}


class BatchNormalization(BaseLayer):
    """
    Normaliza por mini-batch. Mantiene running_mean/running_var para
    inferencia.

    Parámetros entrenables: 'gamma', 'beta'.
    Estado no entrenable: 'running_mean', 'running_var'.
    """

    trainable = True

    def __init__(
        self,
        n_features: Optional[int] = None,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
    ):
        self.n_features = n_features
        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.running_mean: Optional[np.ndarray] = None
        self.running_var: Optional[np.ndarray] = None
        self._built = False

        if n_features is not None:
            self.build((None, n_features))

    def build(self, input_shape):
        n = input_shape[-1]
        if n is None:
            raise ValueError("BatchNormalization: features indefinido en build().")
        self.n_features = n
        self.gamma = np.ones((1, n))
        self.beta = np.zeros((1, n))
        self.running_mean = np.zeros((1, n))
        self.running_var = np.ones((1, n))
        self.input_shape = input_shape
        self.output_shape = input_shape
        self._built = True
        return input_shape

    def forward(self, inputs, training=True):
        if not self._built:
            self.build((None, inputs.shape[1]))
        if training:
            mean = inputs.mean(axis=0, keepdims=True)
            var = inputs.var(axis=0, keepdims=True)
            inv_std = 1.0 / np.sqrt(var + self.epsilon)
            x_hat = (inputs - mean) * inv_std

            # Mutación de estadísticas móviles — esto SÍ es estado
            # persistente (como en Keras/PyTorch). No interfiere con
            # reutilización multi-entrada en modo inference.
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )

            out = self.gamma * x_hat + self.beta
            cache = {"x_hat": x_hat, "inv_std": inv_std}
        else:
            x_hat = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_hat + self.beta
            cache = {}
        return out, cache

    def backward(self, d_output, cache):
        x_hat = cache["x_hat"]
        inv_std = cache["inv_std"]
        N = d_output.shape[0]

        dgamma = np.sum(d_output * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(d_output, axis=0, keepdims=True)

        dx_hat = d_output * self.gamma
        dx = (1.0 / N) * inv_std * (
            N * dx_hat
            - np.sum(dx_hat, axis=0, keepdims=True)
            - x_hat * np.sum(dx_hat * x_hat, axis=0, keepdims=True)
        )
        return dx, {"gamma": dgamma, "beta": dbeta}

    def parameters(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def non_trainable_state(self):
        return {"running_mean": self.running_mean, "running_var": self.running_var}

    def get_config(self):
        return {
            "class_name": "BatchNormalization",
            "config": {
                "n_features": self.n_features,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
            },
        }


_LAYER_CLASSES = {
    "Dense": Dense,
    "Layer": Dense,
    "Dropout": Dropout,
    "BatchNormalization": BatchNormalization,
}


def layer_from_config(config: Dict[str, Any]) -> BaseLayer:
    cls = _LAYER_CLASSES[config["class_name"]]
    return cls.from_config(config.get("config", {}))