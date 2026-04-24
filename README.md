# NeuralNetwork - Librería de Deep Learning Vectorizada

Librería de Deep Learning ligera, modular y **completamente vectorizada** desarrollada en Python y NumPy. Pensada para ser **escalable a producción** sin perder claridad pedagógica: API tipo Keras, gradientes verificados numéricamente, persistencia portable y todas las piezas intercambiables.

> v0.3.0 — refactor mayor: optimizadores Adam/RMSprop/AdaGrad, callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau), BatchNormalization, regularización L1/L2, métricas, `compile()`/`fit()`/`evaluate()` estilo Keras, gradient clipping, persistencia en `.npz` y validación numérica del backprop.

## Características Principales

### Arquitectura
* **Vectorización pura.** Todo opera en mini-batches con álgebra matricial de NumPy. Sin bucles a nivel de neurona.
* **API estilo Keras.** `model.add(Dense(...))`, `model.compile(...)`, `model.fit(...)`, `model.evaluate(...)`, `model.predict(...)`, `model.summary()`.
* **Diseño modular.** Layers, Activations, Losses, Optimizers, Initializers, Regularizers, Metrics, Callbacks. Cada pieza es intercambiable y registrable por string (ej. `optimizer="adam"`).

### Capas
* `Dense` (alias de `Layer`) con inicializador y regularizador opcionales.
* `Dropout` invertido (escala 1/(1-rate) para mantener magnitud).
* `BatchNormalization` con estadísticas móviles para inferencia.

### Activaciones
`Sigmoid`, `ReLU`, `LeakyReLU`, `ELU`, `Tanh`, `Softmax`, `Linear` — todas con caching interno para acelerar el backward pass.

### Optimizadores
* `SGD` con momentum y Nesterov.
* `AdaGrad` para gradientes esparsos.
* `RMSprop` con media móvil exponencial.
* `Adam` con bias correction (recomendado por defecto).
* **Gradient clipping** (`clip_norm` y `clip_value`) en todos.

### Pérdidas
`MSE`, `MAE`, `Huber`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`, `SparseCategoricalCrossEntropy`.

### Inicializadores
`HeNormal`, `XavierNormal`, `XavierUniform`, `Zeros`, `Ones`.

### Regularizadores
`L1`, `L2`, `L1L2` aplicables a los kernels de cada capa.

### Métricas
`BinaryAccuracy`, `CategoricalAccuracy`, `SparseCategoricalAccuracy`, `MeanAbsoluteError`, `RootMeanSquaredError`, `R2Score`.

### Callbacks
* `EarlyStopping` con `restore_best_weights`.
* `ModelCheckpoint` (guarda en `.npz` cuando mejora la métrica).
* `ReduceLROnPlateau` ajusta el learning rate al estancarse.
* `History` se inyecta automáticamente en `fit()`.

### Calidad
* **53+ tests** cubriendo capas, optimizadores, losses, métricas, callbacks, persistencia e integración.
* **Gradient check numérico** que valida matemáticamente el backprop.
* `validation_split` y `validation_data` para monitoreo de generalización.

## Tecnologías Utilizadas
* **Python** 3.8+
* **NumPy** (núcleo de cálculo).
* **Unittest** (suite de pruebas).

## Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/elJulioDev/Neural_Network.git
cd neural_network
```

2. **Crear y activar un entorno virtual:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. **Instalar dependencias o instalar como paquete:**
```bash
pip install -r requirements.txt
# o, para desarrollo:
pip install -e .
```

4. **Ejecutar la suite de tests:**
```bash
python -m unittest discover tests
```

5. **Probar el demo XOR:**
```bash
python main.py
```

6. **Probar el demo multiclase con BatchNorm + Dropout + Callbacks:**
```bash
python examples/multiclass_classification.py
```

## Uso Rápido (API moderna estilo Keras)

```python
import numpy as np
from src import (
    NeuralNetwork, Dense, Dropout, BatchNormalization,
    Adam, EarlyStopping,
)

# Datos
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

# Construcción
model = NeuralNetwork()
model.add(Dense(8, input_size=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Configuración
model.compile(
    optimizer=Adam(learning_rate=0.05),
    loss='bce',
    metrics=['accuracy'],
)

# Entrenamiento con callbacks
history = model.fit(
    X, y,
    epochs=500,
    batch_size=4,
    callbacks=[EarlyStopping(monitor='loss', patience=50)],
    verbose=1,
)

# Inferencia
print(model.predict(X))
print(model.predict_classes(X))
```

## Uso con regularización, BatchNorm y validation_split

```python
from src import Dense, Dropout, BatchNormalization, L2, ReduceLROnPlateau

model = NeuralNetwork()
model.add(Dense(64, input_size=20, activation='relu', kernel_regularizer=L2(0.001)))
model.add(BatchNormalization(64))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.001, clip_norm=1.0),
    loss='cce',
    metrics=['categorical_accuracy'],
)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    ],
)
```

## Persistencia

Dos modos:

**Pesos en `.npz` (portable, recomendado para producción):**
```python
model.save_weights('model_weights.npz')

# Nuevo modelo con la MISMA arquitectura:
new_model = NeuralNetwork()
new_model.add(Dense(64, input_size=20, activation='relu'))
new_model.add(Dense(10, activation='softmax'))
new_model.compile(optimizer='adam', loss='cce')
new_model.load_weights('model_weights.npz')
```

**Modelo completo con `pickle` (incluye arquitectura):**
```python
model.save_model('full_model.pkl')
loaded = NeuralNetwork.load_model('full_model.pkl')
```

## Integración en Proyectos Reales (Ej. Django/Flask)

1. **Instalar la librería:**
```bash
pip install git+https://github.com/elJulioDev/neural_network.git
```

2. **Vista Django:**
```python
from django.http import JsonResponse
from src import NeuralNetwork
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'classifier.pkl')
ai_model = NeuralNetwork.load_model(MODEL_PATH)

def predecir_view(request):
    # IMPORTANTE: entrada siempre 2D (Batch, Features)
    datos = np.array([[0, 1, 0.5, 1.2]])  # shape (1, 4)
    pred = ai_model.predict(datos)        # shape (1, n_clases)
    clase = int(np.argmax(pred[0]))
    return JsonResponse({
        'probabilidades': pred[0].tolist(),
        'clase_predicha': clase,
    })
```

## Guía Técnica

### Formato de entrada y salida
Vectores 1D NO se aceptan. Todas las entradas deben ser matrices 2D `(batch_size, n_features)`.

```python
# Incorrecto:
np.array([0, 1])              # (2,)
# Correcto:
np.array([[0, 1]])            # (1, 2)
```

`predict()` siempre devuelve `(batch_size, n_neurons_salida)`. Para escalar:
```python
valor = float(model.predict(X)[0, 0])
```

### Compatibilidad de modelos
Los `.pkl` antiguos (anteriores a v0.3.0) **no son compatibles** porque las clases internas cambiaron. Re-entrena y guarda nuevamente. **Recomendación:** usa `save_weights()`/`load_weights()` en `.npz` para portabilidad entre versiones (sólo dependes de la arquitectura).

### Gradient clipping
Útil cuando tu loss explota o las redes son profundas:
```python
optimizer = Adam(learning_rate=0.001, clip_norm=1.0)
# o:
optimizer = SGD(learning_rate=0.01, clip_value=5.0)
```

### Activaciones por string
```python
Dense(64, input_size=10, activation='relu')             # válido
Dense(64, input_size=10, activation=LeakyReLU(0.1))     # también válido
```

## Estructura del Proyecto
```text
neural_network/
├── src/                            # Código fuente
│   ├── __init__.py                 # Exports públicos
│   ├── activations.py              # Sigmoid, ReLU, LeakyReLU, ELU, Tanh, Softmax, Linear
│   ├── callbacks.py                # EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, History
│   ├── initializers.py             # He, Xavier (Normal/Uniform), Zeros, Ones
│   ├── layer.py                    # Dense, Dropout, BatchNormalization
│   ├── losses.py                   # MSE, MAE, Huber, BCE, CCE, SparseCCE
│   ├── metrics.py                  # Accuracy, R2, MAE, RMSE
│   ├── neural_network.py           # Modelo Sequential principal
│   ├── optimizers.py               # SGD, AdaGrad, RMSprop, Adam (+ clipping)
│   ├── regularizers.py             # L1, L2, L1L2
│   └── utils.py                    # train_test_split, to_categorical, normalize, etc.
├── tests/                          # Suite de tests (53+)
│   ├── __init__.py
│   ├── test_gradient_check.py      # Validación numérica del backprop
│   ├── test_layer.py               # Dense, Dropout, BatchNorm
│   ├── test_losses.py              # Losses y métricas
│   ├── test_model.py               # Integración: XOR, persistencia, callbacks
│   └── test_optimizers.py          # SGD, Adam, RMSprop, AdaGrad, clipping
├── examples/
│   └── multiclass_classification.py
├── main.py                         # Demo XOR
├── requirements.txt
├── setup.py
└── README.md
```

## Licencia
Proyecto de uso educativo y personal. Distribuido bajo la licencia MIT.