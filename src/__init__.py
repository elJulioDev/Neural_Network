"""NeuralNetwork v0.4 — librería de Deep Learning vectorizada."""
from .neural_network import NeuralNetwork

from .layer import (
    BaseLayer, Layer, Dense, Dropout, BatchNormalization, layer_from_config,
)

from .activations import (
    Activation, Sigmoid, ReLU, LeakyReLU, ELU, Tanh, Softmax, Linear,
    get_activation,
)

from .losses import (
    Loss, MSE, MAE, Huber,
    BinaryCrossEntropy, CategoricalCrossEntropy, SparseCategoricalCrossEntropy,
    get_loss,
)

from .optimizers import (
    Optimizer, SGD, AdaGrad, RMSprop, Adam, get_optimizer,
)

from .initializers import (
    Initializer, HeNormal, XavierNormal, XavierUniform, Zeros, Ones,
    get_initializer,
)

from .regularizers import (
    Regularizer, L1, L2, L1L2, get_regularizer,
)

from .metrics import (
    Metric, BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy,
    MeanAbsoluteError, RootMeanSquaredError, R2Score, get_metric,
)

from .callbacks import (
    Callback, History, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)

from .utils import (
    train_test_split, to_categorical, normalize, standardize,
    shuffle_arrays, batch_iterator,
)

__version__ = "0.4.0"

__all__ = [
    "NeuralNetwork",
    "BaseLayer", "Layer", "Dense", "Dropout", "BatchNormalization", "layer_from_config",
    "Activation", "Sigmoid", "ReLU", "LeakyReLU", "ELU", "Tanh", "Softmax", "Linear",
    "get_activation",
    "Loss", "MSE", "MAE", "Huber",
    "BinaryCrossEntropy", "CategoricalCrossEntropy", "SparseCategoricalCrossEntropy",
    "get_loss",
    "Optimizer", "SGD", "AdaGrad", "RMSprop", "Adam", "get_optimizer",
    "Initializer", "HeNormal", "XavierNormal", "XavierUniform", "Zeros", "Ones",
    "get_initializer",
    "Regularizer", "L1", "L2", "L1L2", "get_regularizer",
    "Metric", "BinaryAccuracy", "CategoricalAccuracy", "SparseCategoricalAccuracy",
    "MeanAbsoluteError", "RootMeanSquaredError", "R2Score", "get_metric",
    "Callback", "History", "EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau",
    "train_test_split", "to_categorical", "normalize", "standardize",
    "shuffle_arrays", "batch_iterator",
]