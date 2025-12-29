import numpy as np
from src.neuron import Neuron

# Configuración básica
inputs = np.array([1, 2, 3])
neuron = Neuron(n_input=3)

# Prueba de Forward
print("--- Forward Pass ---")
output = neuron.forward(inputs)
print(f"Predicción inicial: {output}")

# Prueba de Backward (Entrenamiento simple)
print("\n--- Backward Pass ---")
target = 1.0 # Supongamos que queremos que aprenda a dar 1
learning_rate = 0.1
error_grad = 2 * (output - target) # Derivada simple del error cuadrático

neuron.backward(error_grad, learning_rate)
print("Pesos actualizados.")