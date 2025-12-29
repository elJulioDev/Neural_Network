import numpy as np
from src.neural_network import NeuralNetwork

if __name__ == "__main__":
    # --- Datos de entrenamiento (Compuerta XOR) ---
    # Entradas: [0,0], [0,1], [1,0], [1,1]
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    # Salidas esperadas: 0, 1, 1, 0
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    print("--- Inicializando Red Neuronal ---")
    nn = NeuralNetwork()

    # Estructura: 2 entradas -> Capa Oculta (4 neuronas) -> Salida (1 neurona)
    nn.add_layer(num_neurons=4, input_size=2) # Capa oculta
    nn.add_layer(num_neurons=1)               # Capa de salida

    print("--- Iniciando Entrenamiento ---")
    # Entrenamos con una tasa de aprendizaje baja para ver la convergencia
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    print("\n--- Resultados Finales ---")
    predictions = nn.predict(X)
    
    for i in range(len(X)):
        print(f"Entrada: {X[i]} | Predicción: {predictions[i][0]:.4f} | Esperado: {y[i][0]}")