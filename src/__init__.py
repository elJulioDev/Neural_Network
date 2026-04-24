"""
NeuralNetwork - librería de Deep Learning vectorizada.

Exports principales para uso ergonómico:
    from src import NeuralNetwork, Dense, Dropout, Adam, ...
"""
from .neural_network import NeuralNetwork

from .layer import (
    BaseLayer,
    Layer,
    Dense,
    Dropout,
    BatchNormalization,
)

from .activations import (
    Activation,
    Sigmoid,
    ReLU,
    LeakyReLU,
    ELU,
    Tanh,
    Softmax,
    Linear,
    get_activation,
)

from .losses import (
    Loss,
    MSE,
    MAE,
    Huber,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    SparseCategoricalCrossEntropy,
    get_loss,
)

from .optimizers import (
    Optimizer,
    SGD,
    AdaGrad,
    RMSprop,
    Adam,
    get_optimizer,
)

from .initializers import (
    Initializer,
    HeNormal,
    XavierNormal,
    XavierUniform,
    Zeros,
    Ones,
    get_initializer,
)

from .regularizers import (
    Regularizer,
    L1,
    L2,
    L1L2,
    get_regularizer,
)

from .metrics import (
    Metric,
    BinaryAccuracy,
    CategoricalAccuracy,
    SparseCategoricalAccuracy,
    MeanAbsoluteError,
    RootMeanSquaredError,
    R2Score,
    get_metric,
)

from .callbacks import (
    Callback,
    History,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from .utils import (
    train_test_split,
    to_categorical,
    normalize,
    standardize,
    shuffle_arrays,
    batch_iterator,
)

__version__ = "0.3.0"

__all__ = [
    "NeuralNetwork",
    # Layers
    "BaseLayer", "Layer", "Dense", "Dropout", "BatchNormalization",
    # Activations
    "Activation", "Sigmoid", "ReLU", "LeakyReLU", "ELU", "Tanh", "Softmax", "Linear",
    "get_activation",
    # Losses
    "Loss", "MSE", "MAE", "Huber",
    "BinaryCrossEntropy", "CategoricalCrossEntropy", "SparseCategoricalCrossEntropy",
    "get_loss",
    # Optimizers
    "Optimizer", "SGD", "AdaGrad", "RMSprop", "Adam", "get_optimizer",
    # Initializers
    "Initializer", "HeNormal", "XavierNormal", "XavierUniform", "Zeros", "Ones",
    "get_initializer",
    # Regularizers
    "Regularizer", "L1", "L2", "L1L2", "get_regularizer",
    # Metrics
    "Metric", "BinaryAccuracy", "CategoricalAccuracy", "SparseCategoricalAccuracy",
    "MeanAbsoluteError", "RootMeanSquaredError", "R2Score", "get_metric",
    # Callbacks
    "Callback", "History", "EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau",
    # Utils
    "train_test_split", "to_categorical", "normalize", "standardize",
    "shuffle_arrays", "batch_iterator",
]