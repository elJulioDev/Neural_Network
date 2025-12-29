import numpy as np
import pickle
from .layer import Layer
from .activations import Sigmoid  
from .losses import MSE

class NeuralNetwork:
    def __init__(self, loss_function=MSE()):
        self.layers = []
        self.loss_list = []
        self.loss_function = loss_function
    
    def add_layer(self, num_neurons, input_size=None, activation=None):
        if activation is None:
            activation = Sigmoid()
            
        if not self.layers:
            if input_size is None:
                raise ValueError("Debes definir input_size para la primera capa.")
            self.layers.append(Layer(num_neurons, input_size, activation))
        else:
            previous_output_size = len(self.layers[-1].neurons)
            self.layers.append(Layer(num_neurons, previous_output_size, activation))
          
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, loss_gradient, learning_rate):
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, x, y, epochs=1000, learning_rate=0.1):
        indices = np.arange(len(x))
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Barajado de datos (Shuffle)
            np.random.shuffle(indices)
            
            for i in indices:
                # 1. Forward
                output = self.forward(x[i])
                
                # 2. Calcular Loss
                epoch_loss += self.loss_function.calculate(output, y[i])
                
                # 3. Calcular gradiente
                loss_gradient = self.loss_function.derivative(output, y[i])
                
                # 4. Backward
                self.backward(loss_gradient, learning_rate)
            
            # --- ESTAS LÍNEAS FALTABAN ---
            # Calcular promedio del error en este epoch
            epoch_loss /= len(x)
            self.loss_list.append(epoch_loss)
            
            # Imprimir progreso cada 100 épocas
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {epoch_loss:.6f}")

    def predict(self, x):
        predictions = []
        for i in range(len(x)):
            predictions.append(self.forward(x[i]))
        return np.array(predictions)
    
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Modelo guardado en {filename}")

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)