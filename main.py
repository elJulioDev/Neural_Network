# Nuevo main.py de ejemplo
import numpy as np
from src.neural_network import NeuralNetwork
from src.activations import LeakyReLU, Sigmoid
from src.losses import BinaryCrossEntropy
from src.optimizers import SGD

if __name__ == "__main__":
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    # Definimos optimizador con Momentum (Acelera el entrenamiento)
    optimizer = SGD(learning_rate=0.1, momentum=0.9)
    
    nn = NeuralNetwork(loss_function=BinaryCrossEntropy(), optimizer=optimizer)
    
    # Arquitectura
    nn.add_layer(num_neurons=4, input_size=2, activation=LeakyReLU())
    nn.add_layer(num_neurons=1, activation=Sigmoid())
    
    # Entrenar (Batch size 4 es todo el dataset en este caso pequeño)
    nn.train(X, y, epochs=100000, batch_size=4)
    
    print(nn.predict(X))