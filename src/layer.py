"""
Capas de la red neuronal.

Layer (alias Dense): capa totalmente conectada con regularización
opcional e inicializador configurable.
Dropout: regularización estocástica.
BatchNormalization: normaliza por lotes para acelerar entrenamiento
y permitir learning rates más altos.
"""
from typing import Optional, List
import numpy as np

from .activations import Activation, get_activation
from .initializers import Initializer, get_initializer
from .regularizers import Regularizer, get_regularizer


class BaseLayer:
    """Interfaz común. Subclases definen forward/backward."""

    n_neurons: int = 0
    trainable: bool = False

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> List[np.ndarray]:
        """Lista de parámetros entrenables. Vacío por defecto."""
        return []

    def set_params(self, params: List[np.ndarray]) -> None:
        """Restaura parámetros desde lista (orden importa)."""
        pass

    def regularization_loss(self) -> float:
        """Componente de la loss aportado por regularización."""
        return 0.0


class Layer(BaseLayer):
    """
    Capa totalmente conectada (Dense).

    Args:
        n_neurons: número de neuronas de salida.
        input_size: tamaño de entrada.
        activation: instancia de Activation o string ('relu', 'sigmoid', etc).
        kernel_initializer: estrategia de init para pesos.
        bias_initializer: estrategia de init para biases.
        kernel_regularizer: regularizador opcional sobre pesos (L1/L2/L1L2).
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
        self.input_size = input_size
        self.activation: Activation = get_activation(activation)
        self._kernel_initializer = get_initializer(kernel_initializer)
        self._bias_initializer = get_initializer(bias_initializer)
        self.kernel_regularizer = get_regularizer(kernel_regularizer)

        # Si conocemos input_size, construimos pesos ahora.
        # Si no, se construirán vía build() al añadirla al modelo.
        self.weights: Optional[np.ndarray] = None
        self.biases: Optional[np.ndarray] = None
        if input_size is not None:
            self.build(input_size)

        # Cache para backward
        self.inputs: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.dweights: Optional[np.ndarray] = None
        self.dbiases: Optional[np.ndarray] = None

    def build(self, input_size: int) -> None:
        """Construye los pesos. Se llama tarde si input_size no estaba definido."""
        self.input_size = input_size
        self.weights = self._kernel_initializer((input_size, self.n_neurons))
        self.biases = self._bias_initializer((1, self.n_neurons))

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if self.weights is None:
            self.build(inputs.shape[1])
        self.inputs = inputs
        self.z = inputs @ self.weights + self.biases
        return self.activation.forward(self.z)

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        d_activation = d_output * self.activation.derivative(self.z)

        self.dweights = self.inputs.T @ d_activation
        self.dbiases = np.sum(d_activation, axis=0, keepdims=True)

        # Sumar gradiente de regularización si aplica
        if self.kernel_regularizer is not None:
            self.dweights += self.kernel_regularizer.gradient(self.weights)

        return d_activation @ self.weights.T

    def get_params(self) -> List[np.ndarray]:
        return [self.weights, self.biases]

    def set_params(self, params: List[np.ndarray]) -> None:
        self.weights, self.biases = params[0], params[1]

    def regularization_loss(self) -> float:
        if self.kernel_regularizer is None:
            return 0.0
        return self.kernel_regularizer.loss(self.weights)


# Alias estilo Keras
Dense = Layer


class Dropout(BaseLayer):
    """
    Dropout invertido. Durante training apaga aleatoriamente neuronas;
    en inferencia es identidad. La escala 1/(1-rate) mantiene magnitud.
    """

    trainable = False

    def __init__(self, rate: float):
        if not 0.0 <= rate < 1.0:
            raise ValueError("rate debe estar en [0, 1).")
        self.rate = rate
        self.mask: Optional[np.ndarray] = None
        self.n_neurons = 0

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.rate == 0.0:
            return inputs
        keep_prob = 1.0 - self.rate
        self.mask = (np.random.rand(*inputs.shape) < keep_prob) / keep_prob
        return inputs * self.mask

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return d_output
        return d_output * self.mask


class BatchNormalization(BaseLayer):
    """
    Normaliza activaciones por mini-batch, escalando con gamma y beta
    aprendibles. Reduce internal covariate shift y acelera el
    entrenamiento, permitiendo learning rates más altos.

    Mantiene running mean/var para inferencia.
    """

    trainable = True

    def __init__(self, n_features: int, momentum: float = 0.9, epsilon: float = 1e-5):
        self.n_features = n_features
        self.n_neurons = n_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Parámetros aprendibles
        self.gamma = np.ones((1, n_features))
        self.beta = np.zeros((1, n_features))

        # Estadísticas para inferencia
        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))

        # Gradientes
        self.dgamma: Optional[np.ndarray] = None
        self.dbeta: Optional[np.ndarray] = None

        # Cache para backward
        self._cache = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mean = inputs.mean(axis=0, keepdims=True)
            var = inputs.var(axis=0, keepdims=True)
            inv_std = 1.0 / np.sqrt(var + self.epsilon)
            x_hat = (inputs - mean) * inv_std

            # Actualizar estadísticas móviles
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )

            self._cache = (inputs, mean, inv_std, x_hat)
            return self.gamma * x_hat + self.beta

        # Inferencia: usa estadísticas acumuladas
        x_hat = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self.gamma * x_hat + self.beta

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        inputs, mean, inv_std, x_hat = self._cache
        N = inputs.shape[0]

        self.dgamma = np.sum(d_output * x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(d_output, axis=0, keepdims=True)

        dx_hat = d_output * self.gamma
        # Derivada vectorizada estándar de BatchNorm
        dx = (1.0 / N) * inv_std * (
            N * dx_hat
            - np.sum(dx_hat, axis=0, keepdims=True)
            - x_hat * np.sum(dx_hat * x_hat, axis=0, keepdims=True)
        )
        return dx

    def get_params(self) -> List[np.ndarray]:
        return [self.gamma, self.beta, self.running_mean, self.running_var]

    def set_params(self, params: List[np.ndarray]) -> None:
        self.gamma, self.beta, self.running_mean, self.running_var = params

    # Para que Optimizer la trate uniformemente
    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, value):
        self.gamma = value

    @property
    def biases(self):
        return self.beta

    @biases.setter
    def biases(self, value):
        self.beta = value

    @property
    def dweights(self):
        return self.dgamma

    @property
    def dbiases(self):
        return self.dbeta