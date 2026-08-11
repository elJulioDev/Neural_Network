"""
Layers with external state and generic parameter API.

Key design:

1. `forward(inputs, training) -> (output, cache)`: no longer mutates `self`.
   The cache is a dict with everything `backward` needs. This allows
   reusing the same layer on two inputs (siamese networks) without the
   second overwriting the cache of the first.

2. `backward(d_output, cache) -> (d_input, gradients_dict)`: returns
   gradients in a dictionary named by parameter. The optimizer no longer
   looks for hardcoded `weights`/`biases` attributes.

3. `parameters() -> Dict[str, ndarray]`: the optimizer receives a generic
   dictionary. A layer can have 1, 2, 3 or N different parameters without
   touching the optimizer code.

4. `build(input_shape) -> output_shape`: layers declare their output shape
   given an input shape. The model propagates shapes before training, so
   dimensional incompatibilities are detected in `compile()` rather than
   in the first matrix multiplication.

5. `get_config() / from_config()`: JSON serialization without pickle.

Layers: Dense (alias Layer), Dropout, BatchNormalization.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .activations import Activation, get_activation
from .initializers import get_initializer
from .regularizers import Regularizer, get_regularizer

Cache = Dict[str, Any]
ParamDict = Dict[str, np.ndarray]


class BaseLayer:
    """
    Common interface.

    Contract:
    - `build(input_shape)` is called ONCE before the first forward.
      May be called with `input_shape=None` if no weights are needed
      (Dropout). Returns the output shape.
    - `forward` and `backward` are ideally pure with respect to trainable
      state. For non-trainable statistics (running_mean in BatchNorm),
      mutation is allowed.
    - `parameters()` returns a dict of REFERENCES (not copies) so the
      optimizer can modify in-place.
    """

    trainable: bool = False
    # Input/output shape (fixed in build). (None, features).
    input_shape: Optional[Tuple[Optional[int], int]] = None
    output_shape: Optional[Tuple[Optional[int], int]] = None

    def build(self, input_shape: Tuple[Optional[int], int]) -> Tuple[Optional[int], int]:
        self.input_shape = input_shape
        self.output_shape = input_shape  # default: identity
        return self.output_shape

    def forward(self, inputs: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Cache]:
        raise NotImplementedError

    def backward(self, d_output: np.ndarray, cache: Cache) -> Tuple[np.ndarray, ParamDict]:
        raise NotImplementedError

    def parameters(self) -> ParamDict:
        """TRAINABLE parameters (dict of references)."""
        return {}

    def non_trainable_state(self) -> ParamDict:
        """Persistent non-trainable state (e.g. running_mean)."""
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
    Fully connected layer (Dense).

    Trainable parameters: 'weights', 'biases'.

    Args:
        n_neurons: output size.
        input_size: optional. If omitted, inferred in build().
        activation: instance, string ('relu', 'sigmoid'...) or config dict.
        kernel_initializer: initialization strategy for weights.
        bias_initializer: initialization strategy for biases.
        kernel_regularizer: optional regularizer on weights.
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
                f"{type(self).__name__}: input_shape must have a defined feature dimension."
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
        # Fail-fast shape validation
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"{type(self).__name__}: expected features={self.weights.shape[0]}, "
                f"received {inputs.shape[1]}."
            )
        z = inputs @ self.weights + self.biases
        output, act_cache = self.activation.forward(z)
        cache = {"inputs": inputs, "z": z, "activation_cache": act_cache}
        return output, cache

    def backward(self, d_output, cache):
        # Propagate through activation first (using the activation cache)
        d_z = self.activation.backward(d_output, cache["activation_cache"])
        inputs = cache["inputs"]

        dweights = inputs.T @ d_z
        dbiases = np.sum(d_z, axis=0, keepdims=True)

        # Regularization on weights (not on biases)
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


# Keras-style alias
Dense = Layer


class Dropout(BaseLayer):
    """
    Inverted dropout (scales by 1/keep_prob during training). No trainable
    parameters. Identity at inference.
    """

    trainable = False

    def __init__(self, rate: float):
        if not 0.0 <= rate < 1.0:
            raise ValueError("rate must be in [0, 1).")
        self.rate = rate

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        return input_shape

    def forward(self, inputs, training=True):
        if not training or self.rate == 0.0:
            return inputs, {"mask": None}
        keep_prob = 1.0 - self.rate
        mask = (np.random.default_rng().random(inputs.shape) < keep_prob) / keep_prob
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
    Mini-batch normalization. Maintains running_mean/running_var for
    inference.

    Trainable parameters: 'gamma', 'beta'.
    Non-trainable state: 'running_mean', 'running_var'.
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
            raise ValueError("BatchNormalization: features undefined in build().")
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

            # Mutation of running statistics — this IS persistent state
            # (as in Keras/PyTorch). It does not interfere with
            # multi-input reuse in inference mode.
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
    class_name = config.get("class_name")
    if class_name not in _LAYER_CLASSES:
        raise ValueError(f"Unknown layer: {class_name}. Options: {list(_LAYER_CLASSES.keys())}")
    cls = _LAYER_CLASSES[class_name]
    return cls.from_config(config.get("config", {}))
