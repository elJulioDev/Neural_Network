# NeuralNetwork — Librería de Deep Learning Vectorizada

Librería de Deep Learning ligera, modular y **completamente vectorizada** en Python y NumPy. Pensada para producción: API estilo Keras, gradientes verificados numéricamente, persistencia portable sin pickle, caches externos para arquitecturas avanzadas.

> **v0.4.0 — refactor arquitectónico.** Resuelve los 5 riesgos estructurales de v0.3: acoplamiento matemático oculto (Softmax↔CCE), estado mutable en capas (bloqueaba redes siamesas), serialización pickle-frágil, validación de shapes tardía, y fugas de abstracción en optimizadores. Ver `CHANGELOG` al final.

## Principios de Diseño

1. **Sin acoplamiento matemático oculto.** Softmax implementa el Jacobiano completo; CCE/BCE aceptan `from_logits=True` para el atajo estable. Nada "asume en silencio" quién está antes.
2. **Capas puras respecto al estado entrenable.** `forward(x) -> (output, cache)`, `backward(d_output, cache) -> (d_input, grads)`. Una misma capa procesa N entradas en paralelo (siamesas, triplet loss) sin corromperse.
3. **Interfaz genérica entre capas y optimizadores.** Las capas declaran `parameters() -> Dict[str, ndarray]` con cualquier nombre/cantidad. El optimizer no conoce `weights`/`biases` hardcodeados.
4. **Fail-fast en shapes.** `build()` propaga dimensiones por toda la red en `compile()`, no en el primer `fit()`.
5. **Persistencia portable.** `save(dir)` produce `topology.json` (sin pickle, legible, inspeccionable) + `weights.npz` (estándar NumPy). Sobrevive refactors.

## Características

### Capas
`Dense` (alias `Layer`), `Dropout`, `BatchNormalization`.

### Activaciones
`Sigmoid`, `ReLU`, `LeakyReLU`, `ELU`, `Tanh`, `Softmax` (Jacobiano completo), `Linear`.

### Optimizadores
`SGD` (con momentum y Nesterov), `AdaGrad`, `RMSprop`, `Adam`. Todos con `clip_norm` y `clip_value`.

### Pérdidas
`MSE`, `MAE`, `Huber`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`, `SparseCategoricalCrossEntropy`. Las cross-entropies aceptan `from_logits`.

### Inicializadores
`HeNormal`, `XavierNormal`, `XavierUniform`, `Zeros`, `Ones`.

### Regularizadores
`L1`, `L2`, `L1L2` aplicables a los kernels.

### Métricas
`BinaryAccuracy`, `CategoricalAccuracy`, `SparseCategoricalAccuracy`, `MeanAbsoluteError`, `RootMeanSquaredError`, `R2Score`.

### Callbacks
`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `History`.

### Calidad
* **68 tests** cubriendo capas, optimizadores, losses, métricas, callbacks, persistencia, state isolation e integración.
* **Gradient check numérico** que valida backprop incluido el path Softmax+CCE con Jacobiano real.
* Demo de red siamesa que verifica state isolation con pesos compartidos.

## Instalación

```bash
git clone https://github.com/elJulioDev/Neural_Network.git
cd neural_network
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -e .
python -m unittest discover tests -v
python main.py
python examples/multiclass_classification.py
python examples/siamese_network.py
```

## Uso Rápido (API moderna estilo Keras)

```python
import numpy as np
from src import (
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
    loss=BinaryCrossEntropy(from_logits=True),              # path estable
)

model.fit(X, y, epochs=500, batch_size=4,
          callbacks=[EarlyStopping(monitor='loss', patience=50)],
          verbose=1)

logits = model.predict(X)
probs = 1.0 / (1.0 + np.exp(-logits))
```

## `from_logits` — por qué importa

**Problema en v0.3:** la derivada de `CategoricalCrossEntropy` asumía que la capa anterior era `Softmax`. Si terminabas con `Linear` o `ReLU` y seguías usando CCE, no había error — simplemente los gradientes salían mal y la red no convergía.

**Solución en v0.4:**

* **`from_logits=True`** (recomendado): capa final `Linear`, la loss aplica softmax/sigmoid internamente y usa el atajo estable `(pred - y) / N`.
* **`from_logits=False`**: la capa anterior puede ser cualquier activación. Softmax propaga su Jacobiano real completo — matemáticamente correcto con cualquier loss.

Patrón recomendado para producción:

```python
# Clasificación binaria
model.add(Dense(1, activation='linear'))
model.compile(loss=BinaryCrossEntropy(from_logits=True), ...)
# Al inferir:
logits = model.predict(X)
probs  = 1 / (1 + np.exp(-logits))

# Clasificación multiclase
model.add(Dense(n_classes, activation='linear'))
model.compile(loss=CategoricalCrossEntropy(from_logits=True), ...)
# Al inferir:
logits = model.predict(X)
ex = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = ex / ex.sum(axis=1, keepdims=True)
```

## Persistencia Portable (recomendada)

```python
model.save('my_model/')
# Produce:
#   my_model/topology.json   ← arquitectura + optimizer + loss (legible)
#   my_model/weights.npz     ← parámetros + estado BatchNorm

loaded = NeuralNetwork.load('my_model/')
```

`topology.json` es inspeccionable, no ejecutable, y sobrevive a refactorizaciones internas. NO uses `save_model()` (pickle) en producción.

## Redes con capas compartidas (ej. siamesas)

Una misma instancia de capa puede procesar dos inputs distintos sin corromperse (en v0.3 era imposible). Ver `examples/siamese_network.py`.

```python
from src.layer import Dense
layer = Dense(4, 3, activation='relu')

out1, cache1 = layer.forward(x1)
out2, cache2 = layer.forward(x2)              # NO pisotea cache1

d_in1, grads1 = layer.backward(dL1, cache1)   # usa cache1 — correcto
d_in2, grads2 = layer.backward(dL2, cache2)   # usa cache2 — correcto
```

## Fail-Fast en Shapes

Los mismatches dimensionales se detectan al construir/compilar, no al entrenar:

```python
model = NeuralNetwork()
model.add(Dense(4, input_size=3, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='cce')

# X con 10 features en vez de 3 → ValueError inmediato
model.fit(np.random.randn(5, 10), ...)
```

## Ejemplo Producción con BatchNorm + Dropout + Callbacks

```python
from src import (
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

## Integración en Django/Flask

```python
from src import NeuralNetwork
import numpy as np

ai_model = NeuralNetwork.load('/path/to/production_model/')

def predict_view(request):
    features = np.array([[...]])             # shape (1, n_features)
    logits = ai_model.predict(features)
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    return JsonResponse({'probabilities': probs[0].tolist()})
```

## Estructura del Proyecto

```text
neural_network/
├── src/
│   ├── __init__.py
│   ├── activations.py          # Stateless: forward -> (out, cache)
│   ├── callbacks.py
│   ├── initializers.py         # Con get_config para JSON
│   ├── layer.py                # Dense, Dropout, BatchNorm; parameters() dict
│   ├── losses.py               # from_logits en CCE/BCE
│   ├── metrics.py
│   ├── neural_network.py       # Gestión externa de caches, build(), save/load
│   ├── optimizers.py           # Interfaz genérica (layer_id, name, param, grad)
│   ├── regularizers.py
│   └── utils.py
├── tests/                      # 68 tests
│   ├── test_activations.py     # Stateless, Softmax Jacobiano, roundtrip config
│   ├── test_gradient_check.py  # Valida backprop numéricamente
│   ├── test_layer.py           # Including state isolation test
│   ├── test_losses.py          # Including from_logits path
│   ├── test_model.py           # Integración + persistencia JSON+NPZ
│   └── test_optimizers.py      # Interfaz genérica
├── examples/
│   ├── multiclass_classification.py
│   └── siamese_network.py      # Demuestra state isolation
├── main.py                     # Demo XOR
├── requirements.txt
├── setup.py
└── README.md
```

## CHANGELOG v0.3.0 → v0.4.0

**Cambios estructurales (BREAKING):**
- Capas: `forward(x) -> (output, cache)` y `backward(d_output, cache) -> (d_input, grads_dict)`. Ya no se almacena cache en `self`. **Migración:** si tenías código usando `model.forward()` directamente, ahora obtienes sólo la salida; para depurar el pipeline completo usa `model._forward()` que retorna `(output, caches)`.
- Capas: `parameters() -> Dict[str, ndarray]` reemplaza a `get_params()`. Los optimizadores ya no acceden a `layer.weights`/`layer.biases`.
- Optimizadores: `apply_gradients(list_of_tuples)` reemplaza a `update(layers)`. Aceptan cualquier nombre de parámetro.
- Losses: `BinaryCrossEntropy` y `CategoricalCrossEntropy` aceptan `from_logits`.
- Softmax: backward ahora implementa el Jacobiano completo (no "return 1").

**Nuevo:**
- `NeuralNetwork.build(input_shape)` propaga shapes y valida fail-fast.
- `NeuralNetwork.save(dir) / load(dir)` → topology.json + weights.npz (sin pickle).
- `to_json() / from_json()` en modelo y componentes.
- `examples/siamese_network.py` verifica state isolation con pesos compartidos.
- Tests: 51 → 68.

**Arreglado:**
- Acoplamiento matemático oculto CCE↔Softmax (issue #1).
- Estado mutable que rompía arquitecturas multi-entrada (issue #2).
- Serialización frágil con pickle (issue #3).
- Validación tardía de shapes (issue #4).
- Optimizador dependiente de `weights`/`biases` hardcodeados (issue #5).

**Migración rápida desde v0.3:**
```python
# v0.3
model.save_model('model.pkl')
NeuralNetwork.load_model('model.pkl')

# v0.4 (recomendado)
model.save('model/')
NeuralNetwork.load('model/')
```

```python
# v0.3 — riesgo oculto de CCE asumiendo Softmax
model.add(Dense(10, activation='softmax'))
model.compile(loss='cce', ...)

# v0.4 — camino recomendado, numéricamente estable
model.add(Dense(10, activation='linear'))
model.compile(loss=CategoricalCrossEntropy(from_logits=True), ...)
```

## Licencia
Proyecto de uso educativo y personal. Distribuido bajo la licencia MIT.