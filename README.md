# NeuralNetwork - Librería de Deep Learning Vectorizada

Este proyecto es una librería de Deep Learning ligera, modular y **completamente vectorizada** desarrollada en Python y NumPy. A diferencia de implementaciones educativas básicas, **NeuralNetwork** utiliza operaciones matriciales para un rendimiento superior, implementando desde cero algoritmos de retropropagación (Backpropagation), optimizadores con momentum y diversas funciones de activación.

Está diseñada para ser escalable, permitiendo crear arquitecturas profundas para resolver problemas de clasificación binaria, multiclase y regresión.

## Características Principales

* **Arquitectura Vectorizada:**
    * **Alto Rendimiento:** Eliminación de bucles a nivel de neurona. Las capas (`Layer`) procesan lotes de datos (batches) utilizando álgebra matricial eficiente.
    * **Diseño Modular:** Componentes desacoplados para Capas, Activaciones, Pérdidas y Optimizadores.

* **Sistema de Optimizadores:**
    * **SGD con Momentum:** Implementación de Descenso de Gradiente Estocástico con término de momento para acelerar la convergencia y evitar mínimos locales.
    * **Gestión de Hiperparámetros:** Control granular del *learning rate* y *momentum*.

* **Flexibilidad Arquitectónica:**
    * **Activaciones:** `Sigmoid`, `ReLU`, `LeakyReLU` (con prevención de neuronas muertas), `Softmax` (para multiclase) y `Linear` (para regresión).
    * **Capas Especiales:** Soporte para **Dropout** para regularización y prevención de overfitting.
    * **Funciones de Pérdida:** `MSE` (Error Cuadrático Medio), `BinaryCrossEntropy` y `CategoricalCrossEntropy`.

* **Entrenamiento Profesional:**
    * **Mini-Batch Training:** Procesamiento de datos en lotes para mayor estabilidad y velocidad.
    * **Inicialización de He:** Pesos inicializados inteligentemente para redes profundas.
    * **Persistencia:** Guardado y carga de modelos entrenados (`pickle`).

## Tecnologías Utilizadas
* **Lenguaje:** Python 3.10+
* **Cálculo Numérico:** NumPy (Operaciones matriciales y álgebra lineal).
* **Testing:** Unittest (Cobertura de capas, optimizadores y pérdidas).

## Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/elJulioDev/Neural_Network.git
cd neural_network
```

2. **Instalar dependencias:** Puedes instalarlo como un paquete local editable o instalar las dependencias directamente:
```bash
pip install -r requirements.txt
# O alternativamente para desarrollo:
pip install -e .
```

3. **Crear y activar un entorno virtual:**
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

4. **Ejecutar Pruebas Unitarias:** Para asegurar que toda la matemática base funciona correctamente:
```bash
python -m unittest discover tests
```

5. **Ejecutar el ejemplo (XOR):** Entrena la red para resolver la compuerta lógica XOR:
```bash
python main.py
```

## Uso del Sistema
La librería está diseñada para ser intuitiva. Aquí tienes un ejemplo de cómo configurar una red para clasificación:
```python
import numpy as np
from src.neural_network import NeuralNetwork
from src.activations import LeakyReLU, Sigmoid
from src.losses import BinaryCrossEntropy
from src.optimizers import SGD

# 1. Datos (XOR)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 2. Configurar Optimizador (NUEVO: Momentum incluido)
optimizer = SGD(learning_rate=0.1, momentum=0.9)

# 3. Inicializar Red
nn = NeuralNetwork(loss_function=BinaryCrossEntropy(), optimizer=optimizer)

# 4. Definir Arquitectura
# Capa oculta: 2 entradas -> 4 neuronas (LeakyReLU)
nn.add_layer(num_neurons=4, input_size=2, activation=LeakyReLU())
# Capa salida: 1 neurona (Sigmoid)
nn.add_layer(num_neurons=1, activation=Sigmoid())

# 5. Entrenar (NUEVO: Soporte para batch_size)
nn.train(X, y, epochs=5000, batch_size=4)

# 6. Predecir
print(nn.predict(X))
```

## Integración en Proyectos Reales (Ej. Django/Flask)

Gracias a que `NeuralNetwork` es un paquete instalable, puedes integrarlo fácilmente en backends web.

1. **Instalar la librería en tu otro proyecto:**
```bash
# Desde la carpeta de tu proyecto Django
pip install git+[https://github.com/elJulioDev/neural_network.git](https://github.com/elJulioDev/neural_network.git)
```

2. **Ejemplo de uso en una vista de Django (views.py):**
```python
from django.http import JsonResponse
from neural_network import NeuralNetwork
import numpy as np
import os

# CARGA DEL MODELO (Singleton)
# Asegúrate de que este 'xor_model.pkl' haya sido entrenado con la versión vectorizada
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelos', 'xor_model.pkl')
try:
    ai_model = NeuralNetwork.load_model(MODEL_PATH)
except Exception as e:
    # Es buena práctica manejar si el modelo no carga (ej. versiones incompatibles)
    print(f"Error cargando el modelo: {e}")
    ai_model = None

def predecir_view(request):
    if ai_model is None:
        return JsonResponse({'error': 'El modelo no está disponible'}, status=500)

    # 1. Preparar datos
    # La nueva librería EXIGE una matriz 2D: (Batch_Size, Input_Size)
    # Aquí Batch_Size = 1
    datos_entrada = np.array([[0, 1]]) 

    # 2. Inferencia
    # Devuelve un np.array de forma (1, 1)
    prediccion_matriz = ai_model.predict(datos_entrada)

    # 3. Extracción
    # Accedemos a la fila 0, columna 0 para obtener el escalar
    valor_predicho = float(prediccion_matriz[0][0])

    return JsonResponse({
        'input': [0, 1],
        'prediccion': valor_predicho,
        'clase': 1 if valor_predicho > 0.5 else 0
    })
```

## Guía Técnica y Solución de Problemas
Esta sección es crucial para integrar la librería en producción (Django, Flask, FastAPI) y evitar errores comunes.

1. **Formato de Entrada (Input Shapes)**
Debido a la vectorización, la librería es estricta con las dimensiones. No se aceptan vectores 1D.

- **Incorrecto:** `np.array([0, 1])` -> Forma `(2,)` -> Causará error de dimensiones.
- **Correcto:** `np.array([[0, 1]])` -> Forma `(1, 2)` -> Matriz de 1 fila y 2 columnas (Batch de tamaño 1).

2. **Formato de Salida (Output)**
El método `.predict()` siempre devuelve una matriz `(Batch_Size, Neuronas_Salida)`.

```python
pred = model.predict(np.array([[0, 1]]))
# Resultado: array([[ 0.98 ]])

# Para obtener el valor escalar (float):
valor = float(pred[0][0])
```

3. **Compatibilidad de Modelos (.pkl)**
Si actualizaste la librería desde una versión anterior (v0.1.0 o previa), **tus modelos antiguos (.pkl) no funcionarán.**

- Causa: La clase `Neuron` fue eliminada y la estructura interna de `Layer` cambió drásticamente.
- Solución: Debes re-entrenar tus modelos con la nueva versión y guardarlos nuevamente.

## Estructura del Proyecto
```text
neural_network/
├── src/                            # Código fuente (Core)
│   ├── __init__.py
│   ├── activations.py              # Sigmoid, ReLU, Softmax, Linear
│   ├── layer.py                    # Lógica de capas vectorizadas y Dropout
│   ├── losses.py                   # MSE, CrossEntropy (Binaria/Categórica)
│   ├── neural_network.py           # Orquestador y bucle de entrenamiento
│   └── optimizers.py               # Algoritmos de optimización (SGD)
├── tests/                          # Pruebas Unitarias
│   ├── __init__.py
│   ├── test_layer.py               # Test de operaciones matriciales
│   ├── test_losses.py
│   └── test_optimizers.py          # Test de actualizaciones de pesos
├── main.py                         # Script de demostración
├── requirements.txt                # Dependencias
└── README.md                       # Documentación
```

## Licencia
Este proyecto es de uso educativo y personal. Se distribuye bajo la licencia MIT.
