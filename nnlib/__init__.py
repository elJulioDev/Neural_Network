"""NeuralNetwork v1.0 — vectorized Deep Learning library."""
from .activations import (
    ELU,
    Activation,
    LeakyReLU,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    get_activation,
)
from .callbacks import (
    Callback,
    EarlyStopping,
    History,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from .initializers import (
    HeNormal,
    Initializer,
    Ones,
    XavierNormal,
    XavierUniform,
    Zeros,
    get_initializer,
)
from .layer import (
    BaseLayer,
    BatchNormalization,
    Dense,
    Dropout,
    Layer,
    layer_from_config,
)
from .losses import (
    MAE,
    MSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    Huber,
    Loss,
    SparseCategoricalCrossEntropy,
    get_loss,
)
from .metrics import (
    BinaryAccuracy,
    CategoricalAccuracy,
    MeanAbsoluteError,
    Metric,
    R2Score,
    RootMeanSquaredError,
    SparseCategoricalAccuracy,
    get_metric,
)
from .neural_network import NeuralNetwork
from .optimizers import (
    SGD,
    AdaGrad,
    Adam,
    Optimizer,
    RMSprop,
    get_optimizer,
)
from .regularizers import (
    L1,
    L1L2,
    L2,
    Regularizer,
    get_regularizer,
)
from .utils import (
    batch_iterator,
    normalize,
    shuffle_arrays,
    standardize,
    to_categorical,
    train_test_split,
)

__version__ = "1.0.0"

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
