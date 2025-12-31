# NeuralNetwork - Librería de Deep Learning desde Cero

Este proyecto es una librería de Deep Learning ligera y modular desarrollada íntegramente en Python y NumPy. A diferencia de frameworks de alto nivel como TensorFlow o PyTorch, **NeuralNetwork** implementa la matemática de las redes neuronales desde la base, permitiendo una comprensión profunda de los algoritmos de retropropagación (Backpropagation), optimización y funciones de activación.

Está diseñada para ser escalable y educativa, permitiendo la creación de arquitecturas personalizadas para resolver problemas de clasificación binaria y regresión, como el clásico problema XOR.

## Características Principales
* **Arquitectura Modular:**
    * **Diseño Orientado a Objetos:** Separación lógica entre Neuronas, Capas (`Layer`) y el Orquestador (`NeuralNetwork`).
    * **Activaciones Flexibles:** Implementación de funciones `Sigmoid`, `ReLU` y `LeakyReLU` intercambiables por capa.
* **Optimización Matemática Avanzada:**
    * **Inicialización de He:** Inicialización inteligente de pesos para prevenir el desvanecimiento de gradientes en redes profundas.
    * **Funciones de Pérdida (Loss Functions):** Soporte para `MSE` (Error Cuadrático Medio) y `BinaryCrossEntropy` (Entropía Cruzada Binaria).
    * **Prevención de "Dying ReLU":** Implementación de `LeakyReLU` para mantener el flujo de gradientes en valores negativos.
* **Entrenamiento Robusto:**
    * **Stochastic Gradient Descent (SGD):** Optimización de pesos mediante descenso de gradiente.
    * **Data Shuffling:** Barajado automático de datos en cada época para evitar mínimos locales y ciclos repetitivos.
* **Persistencia de Modelos:**
    * Sistema nativo para guardar (`save_model`) y cargar (`load_model`) redes entrenadas utilizando `pickle`.
* **Visualización:**
    * Integración con `matplotlib` para generar curvas de aprendizaje y monitorear la convergencia del error en tiempo real.

## Tecnologías Utilizadas
El proyecto utiliza un stack enfocado en el cálculo numérico y la eficiencia matemática:
* **Lenguaje:** Python 3.10+
* **Cálculo Numérico:** NumPy (Álgebra lineal, operaciones matriciales).
* **Empaquetado:** Setuptools (Estructura de librería instalable).
* **Visualización:** Matplotlib (Gráficos de curvas de pérdida).
* **Testing:** Unittest (Pruebas unitarias para neuronas, capas y pérdidas).

## Pre-requisitos
Asegúrate de tener instalado y configurado lo siguiente:
* Python 3.8 o superior
* Git
* Virtualenv (recomendado)

## Instalación y Configuración
Sigue estos pasos para levantar el proyecto en tu entorno local:

1. **Clonar el repositorio:**
```bash
git clone [https://github.com/elJulioDev/Neural_Network.git](https://github.com/elJulioDev/Neural_Network.git)
cd neural_network
```

2. **Crear y activar un entorno virtual:**
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

3. **Instalar dependencias:** Puedes instalarlo como un paquete local editable o instalar las dependencias directamente:
```bash
pip install -r requirements.txt
# O alternativamente para desarrollo:
pip install -e .
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

# 1. Inicializar la red con función de pérdida
nn = NeuralNetwork(loss_function=BinaryCrossEntropy())

# 2. Definir Arquitectura
# Capa de entrada (2 neuronas) -> Oculta (4 neuronas, LeakyReLU)
nn.add_layer(num_neurons=4, input_size=2, activation=LeakyReLU())
# Capa de salida (1 neurona, Sigmoid)
nn.add_layer(num_neurons=1, activation=Sigmoid())

# 3. Entrenar
nn.train(X, y, epochs=10000, learning_rate=0.1)

# 4. Predecir
predicciones = nn.predict(X)
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

# Cargar el modelo entrenado (Singleton recomendado para producción)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelos', 'xor_model.pkl')
ai_model = NeuralNetwork.load_model(MODEL_PATH)

def predecir_view(request):
    # Suponiendo que recibes datos [0, 1]
    datos_entrada = np.array([[0, 1]]) 

    # Realizar inferencia
    prediccion = ai_model.predict(datos_entrada)

    return JsonResponse({
        'input': [0, 1],
        'prediccion': float(prediccion[0][0]),
        'clase': 1 if prediccion > 0.5 else 0
    })
```

## Estructura del Proyecto
```text
neural_network/
├── src/                            # Código fuente de la librería
│   ├── __init__.py
│   ├── activations.py              # Funciones (Sigmoid, ReLU, LeakyReLU)
│   ├── layer.py                    # Lógica de capas y conexión de neuronas
│   ├── losses.py                   # Funciones de costo (MSE, CrossEntropy)
│   ├── neural_network.py           # Orquestador principal y bucle de entrenamiento
│   └── neuron.py                   # Lógica base de la neurona (pesos/bias)
├── tests/                          # Pruebas Unitarias
│   ├── __init__.py
│   ├── test_layer.py
│   ├── test_losses.py
│   └── test_neuron.py
├── main.py                         # Script de demostración (Problema XOR)
├── setup.py                        # Configuración de instalación del paquete
├── requirements.txt                # Dependencias
└── .gitignore                      # Archivos ignorados
```

## Licencia
Este proyecto es de uso educativo y personal. Se distribuye bajo la licencia MIT.
