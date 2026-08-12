# NeuralNetwork — Vectorized Deep Learning Library

[![CI](https://github.com/elJulioDev/Neural_Network/actions/workflows/python-app.yml/badge.svg)](https://github.com/elJulioDev/Neural_Network/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Lightweight, modular, and **fully vectorized** Deep Learning library in Python and NumPy. Designed for production: Keras-style API, numerically verified gradients, portable persistence without pickle, external caches for advanced architectures.

> **v1.0.0 — stable release.** Package renamed to `nnlib`, modern packaging with `pyproject.toml`, CI with linting, MIT LICENSE. See [CHANGELOG.md](CHANGELOG.md) for full history.

## Design Principles

1. **No hidden mathematical coupling.** Softmax implements the full Jacobian; CCE/BCE accept `from_logits=True` for the stable shortcut. Nothing "silently assumes" who is before it.
2. **Layers are pure w.r.t. trainable state.** `forward(x) -> (output, cache)`, `backward(d_output, cache) -> (d_input, grads)`. The same layer processes N inputs in parallel (siamese, triplet loss) without corruption.
3. **Generic interface between layers and optimizers.** Layers declare `parameters() -> Dict[str, ndarray]` with any name/quantity. The optimizer doesn't know about hardcoded `weights`/`biases`.
4. **Fail-fast on shapes.** `build()` propagates dimensions through the entire network at `compile()` time, not on the first `fit()`.
5. **Portable persistence.** `save(dir)` produces `topology.json` (no pickle, readable, inspectable) + `weights.npz` (standard NumPy). Survives refactors.

## Features

### Layers
`Dense` (alias `Layer`), `Dropout`, `BatchNormalization`.

### Activations
`Sigmoid`, `ReLU`, `LeakyReLU`, `ELU`, `Tanh`, `Softmax` (full Jacobian), `Linear`.

### Optimizers
`SGD` (with momentum and Nesterov), `AdaGrad`, `RMSprop`, `Adam`. All with `clip_norm` and `clip_value`.

### Losses
`MSE`, `MAE`, `Huber`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`, `SparseCategoricalCrossEntropy`. Cross-entropies accept `from_logits`.

### Initializers
`HeNormal`, `XavierNormal`, `XavierUniform`, `Zeros`, `Ones`.

### Regularizers
`L1`, `L2`, `L1L2` applicable to kernels.

### Metrics
`BinaryAccuracy`, `CategoricalAccuracy`, `SparseCategoricalAccuracy`, `MeanAbsoluteError`, `RootMeanSquaredError`, `R2Score`.

### Callbacks
`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `History`.

### Quality
* **68 tests** covering layers, optimizers, losses, metrics, callbacks, persistence, state isolation, and integration.
* **Numerical gradient check** validating backprop including the Softmax+CCE path with real Jacobian.
* Siamese network demo verifying state isolation with shared weights.

## Installation

```bash
git clone https://github.com/elJulioDev/Neural_Network.git
cd Neural_Network
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -e .
```

For development (includes ruff and matplotlib):

```bash
pip install -e ".[dev]"
```

## Quick Start (Keras-style API)

```python
import numpy as np
from nnlib import (
    NeuralNetwork, Dense, Dropout, BatchNormalization,
    Adam, BinaryCrossEntropy, EarlyStopping,
)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

model = NeuralNetwork()
model.add(Dense(8, input_size=2, activation='relu'))
model.add(Dense(1, activation='linear'))                    # logits

model.compile(
    optimizer=Adam(learning_rate=0.05),
    loss=BinaryCrossEntropy(from_logits=True),              # stable path
)

model.fit(X, y, epochs=500, batch_size=4,
          callbacks=[EarlyStopping(monitor='loss', patience=50)],
          verbose=1)

logits = model.predict(X)
probs = 1.0 / (1.0 + np.exp(-logits))
```

## `from_logits` — why it matters

* **`from_logits=True`** (recommended): final layer `Linear`, the loss applies softmax/sigmoid internally and uses the stable shortcut `(pred - y) / N`.
* **`from_logits=False`**: the previous layer can be any activation. Softmax propagates its full real Jacobian — mathematically correct with any loss.

Recommended production pattern:

```python
# Binary classification
model.add(Dense(1, activation='linear'))
model.compile(loss=BinaryCrossEntropy(from_logits=True), ...)
# At inference:
logits = model.predict(X)
probs  = 1 / (1 + np.exp(-logits))

# Multiclass classification
model.add(Dense(n_classes, activation='linear'))
model.compile(loss=CategoricalCrossEntropy(from_logits=True), ...)
# At inference:
logits = model.predict(X)
ex = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = ex / ex.sum(axis=1, keepdims=True)
```

## Portable Persistence

```python
model.save('my_model/')
# Produces:
#   my_model/topology.json   <- architecture + optimizer + loss (readable)
#   my_model/weights.npz     <- parameters + BatchNorm state

loaded = NeuralNetwork.load('my_model/')
```

`topology.json` is inspectable, not executable, and survives internal refactors.

## Shared-weight networks (e.g. siamese)

The same layer instance can process two different inputs without corruption. See `examples/siamese_network.py`.

```python
from nnlib.layer import Dense
layer = Dense(4, 3, activation='relu')

out1, cache1 = layer.forward(x1)
out2, cache2 = layer.forward(x2)              # does NOT overwrite cache1

d_in1, grads1 = layer.backward(dL1, cache1)   # uses cache1 — correct
d_in2, grads2 = layer.backward(dL2, cache2)   # uses cache2 — correct
```

## Fail-Fast on Shapes

Dimensional mismatches are detected at build/compile time, not during training:

```python
model = NeuralNetwork()
model.add(Dense(4, input_size=3, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='cce')

# X has 10 features instead of 3 -> immediate ValueError
model.fit(np.random.randn(5, 10), ...)
```

## Production Example with BatchNorm + Dropout + Callbacks

```python
from nnlib import (
    NeuralNetwork, Dense, Dropout, BatchNormalization,
    Adam, L2, CategoricalCrossEntropy,
    EarlyStopping, ReduceLROnPlateau,
)

model = NeuralNetwork()
model.add(Dense(64, input_size=20, activation='relu', kernel_regularizer=L2(0.001)))
model.add(BatchNormalization(64))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='linear'))                    # logits

model.compile(
    optimizer=Adam(learning_rate=0.001, clip_norm=1.0),
    loss=CategoricalCrossEntropy(from_logits=True),
)

model.fit(X_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          callbacks=[
              EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
              ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
          ])

model.save('production_model/')
```

## Django/Flask Integration

```python
from nnlib import NeuralNetwork
import numpy as np

ai_model = NeuralNetwork.load('/path/to/production_model/')

def predict_view(request):
    features = np.array([[...]])             # shape (1, n_features)
    logits = ai_model.predict(features)
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    return JsonResponse({'probabilities': probs[0].tolist()})
```

## Project Structure

```text
Neural_Network/
├── nnlib/                        # Main package
│   ├── __init__.py               # Public API re-exports
│   ├── activations.py            # Stateless: forward -> (out, cache)
│   ├── callbacks.py
│   ├── initializers.py           # With get_config for JSON
│   ├── layer.py                  # Dense, Dropout, BatchNorm; parameters() dict
│   ├── losses.py                 # from_logits in CCE/BCE
│   ├── metrics.py
│   ├── neural_network.py         # External cache management, build(), save/load
│   ├── optimizers.py             # Generic interface (layer_id, name, param, grad)
│   ├── regularizers.py
│   └── utils.py
├── tests/                        # 68 tests
│   ├── test_activations.py       # Stateless, Softmax Jacobian, config roundtrip
│   ├── test_gradient_check.py    # Numerical backprop validation
│   ├── test_layer.py             # Including state isolation test
│   ├── test_losses.py            # Including from_logits path
│   ├── test_model.py             # Integration + JSON+NPZ persistence
│   └── test_optimizers.py        # Generic interface
├── examples/
│   ├── multiclass_classification.py
│   └── siamese_network.py        # Demonstrates state isolation
├── main.py                       # XOR demo
├── pyproject.toml                # Modern packaging (PEP 517/518)
├── requirements.txt              # Dev dependencies
├── CHANGELOG.md
├── LICENSE                       # MIT
└── README.md
```

## Running Tests

```bash
python -m unittest discover tests -v
```

## Examples

```bash
python main.py
python examples/multiclass_classification.py
python examples/siamese_network.py
```

## Limitations

* No convolutional or recurrent layers (Conv1D/2D, LSTM, GRU, etc.) — contributions welcome.
* No GPU acceleration — pure NumPy, CPU only.
* Full dataset must fit in memory — no streaming or lazy loading for large datasets.
* No built-in model export to ONNX or other formats.

## License

MIT License. See [LICENSE](LICENSE) for details.
