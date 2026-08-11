# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-08-10

### Added
- Modern packaging with `pyproject.toml` (PEP 517/518).
- `LICENSE` file (MIT).
- Linting configuration with `ruff` in CI.
- `CHANGELOG.md`.

### Changed
- Package renamed from `src` to `nnlib` (imports: `from nnlib import NeuralNetwork`).
- Removed `sys.path.insert` from tests and examples (no longer needed with `pip install -e .`).
- CI updated: Python 3.9/3.10/3.11/3.12 matrix, install with `pip install -e ".[dev]"`.
- `siamese_network.py` moved from `src/` to `examples/`.
- Python 3.8 dropped from support (EOL).

### Removed
- `setup.py` (replaced by `pyproject.toml`).

## [0.4.0] - No tag

### Added
- `NeuralNetwork.build(input_shape)` propagates shapes and validates fail-fast.
- `NeuralNetwork.save(dir) / load(dir)` with topology.json + weights.npz.
- `to_json() / from_json()` on model and components.
- `examples/siamese_network.py` verifies state isolation with shared weights.
- Tests: 51 to 68.

### Changed
- Layers: `forward(x) -> (output, cache)` and `backward(d_output, cache) -> (d_input, grads_dict)`.
- Layers: `parameters() -> Dict[str, ndarray]` replaces `get_params()`.
- Optimizers: `apply_gradients(list_of_tuples)` replaces `update(layers)`.
- Losses: `BinaryCrossEntropy` and `CategoricalCrossEntropy` accept `from_logits`.
- Softmax: backward implements the full Jacobian.

### Fixed
- Hidden mathematical coupling CCE/Softmax.
- Mutable state breaking multi-input architectures.
- Fragile pickle serialization.
- Late shape validation.
- Optimizer dependent on hardcoded `weights`/`biases`.

## [0.3.0] - No tag

### Added
- Keras-style API with `compile()`, `fit()`, `predict()`.
- Adam, RMSprop, AdaGrad optimizers with gradient clipping.
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
- BatchNormalization.

## [0.2.0] - No tag

### Added
- Full vectorization of layers.
- He initialization, LeakyReLU, data shuffling.
- BinaryCrossEntropy loss.

## [0.1.0] - No tag

### Added
- Base structure with Neuron, backpropagation, unit tests.
