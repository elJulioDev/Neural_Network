import numpy as np
from .layer import Layer

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_list = []
    
    # CORRECCIÓN: Cambiado de add_Layer a add_layer (PEP 8 standard)
    # Además, hacemos input_size opcional para capas ocultas/salida
    def add_layer(self, num_neurons, input_size=None):
        if not self.layers:
            # Para la primera capa, input_size es obligatorio
            if input_size is None:
                raise ValueError("Debes definir input_size para la primera capa.")
            self.layers.append(Layer(num_neurons, input_size))
        else:
            # Para las siguientes, tomamos el tamaño de salida de la anterior
            # CORRECCIÓN TYPO: 'previos' -> 'previous'
            previous_output_size = len(self.layers[-1].neurons)
            self.layers.append(Layer(num_neurons, previous_output_size))
          
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, loss_gradient, learning_rate):
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, x, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            loss = 0
            for i in range(len(x)):
                output = self.forward(x[i])
                
                # Calcular Loss (Error Cuadrático Medio)
                loss += np.mean((y[i] - output) ** 2)
                
                # Calcular gradiente del error (Derivada simple del MSE)
                loss_gradient = 2 * (output - y[i])
                
                self.backward(loss_gradient, learning_rate)
            
            loss /= len(x)
            self.loss_list.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss:.6f}")

    def predict(self, x):
        predictions = []
        for i in range(len(x)):
            predictions.append(self.forward(x[i]))
        return np.array(predictions)