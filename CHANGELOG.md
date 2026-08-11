# Changelog

Todos los cambios notables en este proyecto seran documentados en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/), y este proyecto adherise a [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-08-10

### Added
- Empaquetado moderno con `pyproject.toml` (PEP 517/518).
- Archivo `LICENSE` (MIT).
- Configuracion de linting con `ruff` en CI.
- `CHANGELOG.md`.

### Changed
- Paquete renombrado de `src` a `nnlib` (imports: `from nnlib import NeuralNetwork`).
- Eliminados los `sys.path.insert` de tests y examples (ya no son necesarios con `pip install -e .`).
- CI actualizado: matrix Python 3.9/3.10/3.11/3.12, instalacion con `pip install -e ".[dev]"`.
- `siamese_network.py` movido de `src/` a `examples/`.
- Python 3.8 eliminado del soporte (EOL).

### Removed
- `setup.py` (reemplazado por `pyproject.toml`).

## [0.4.0] - Sin tag

### Added
- `NeuralNetwork.build(input_shape)` propaga shapes y valida fail-fast.
- `NeuralNetwork.save(dir) / load(dir)` con topology.json + weights.npz.
- `to_json() / from_json()` en modelo y componentes.
- `examples/siamese_network.py` verifica state isolation con pesos compartidos.
- Tests: 51 a 68.

### Changed
- Capas: `forward(x) -> (output, cache)` y `backward(d_output, cache) -> (d_input, grads_dict)`.
- Capas: `parameters() -> Dict[str, ndarray]` reemplaza a `get_params()`.
- Optimizadores: `apply_gradients(list_of_tuples)` reemplaza a `update(layers)`.
- Losses: `BinaryCrossEntropy` y `CategoricalCrossEntropy` aceptan `from_logits`.
- Softmax: backward implementa el Jacobiano completo.

### Fixed
- Acoplamiento matematico oculto CCE/Softmax.
- Estado mutable que rompia arquitecturas multi-entrada.
- Serializacion fragil con pickle.
- Validacion tardia de shapes.
- Optimizador dependiente de `weights`/`biases` hardcodeados.

## [0.3.0] - Sin tag

### Added
- API estilo Keras con `compile()`, `fit()`, `predict()`.
- Optimizadores Adam, RMSprop, AdaGrad con gradient clipping.
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
- BatchNormalization.

## [0.2.0] - Sin tag

### Added
- Vectorizacion completa de capas.
- Inicializacion He, LeakyReLU, data shuffling.
- BinaryCrossEntropy loss.

## [0.1.0] - Sin tag

### Added
- Estructura base con Neuron, backpropagation, tests unitarios.
