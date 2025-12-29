import numpy as np
from src.layer import Layer

# 1. Crear datos de prueba
inputs = np.array([1.0, 2.0, 3.0])  # 3 Entradas

# 2. Crear una Capa
# 4 neuronas, cada una espera 3 entradas
layer = Layer(num_neurons=4, inputs_size=3) 

# --- Forward Pass ---
print("--- Forward Pass ---")
outputs = layer.forward(inputs)
print(f"Salidas de la capa (4 neuronas): {outputs}")

# --- Backward Pass ---
print("\n--- Backward Pass ---")
# Simulamos un gradiente de error que viene de la siguiente capa (o función de pérdida)
# Debe tener el mismo tamaño que 'outputs' (4)
d_outputs = np.array([0.1, -0.1, 0.0, 0.5]) 
learning_rate = 0.1

d_inputs = layer.backward(d_outputs, learning_rate)
print(f"Gradientes para la capa anterior: {d_inputs}")
print("Pesos de las neuronas actualizados.")