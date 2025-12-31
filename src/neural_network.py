import numpy as np
import pickle
from .layer import Layer, Dropout
from .activations import Sigmoid
from .losses import MSE
from .optimizers import SGD

class NeuralNetwork:
    def __init__(self, loss_function=MSE(), optimizer=None):
        self.layers = []
        self.loss_function = loss_function
        # Por defecto usamos SGD si no se pasa uno
        self.optimizer = optimizer if optimizer else SGD(learning_rate=0.01)
    
    def add_layer(self, num_neurons, input_size=None, activation=None):
        if activation is None:
            activation = Sigmoid()
            
        if not self.layers:
            if input_size is None:
                raise ValueError("Debes definir input_size para la primera capa.")
            self.layers.append(Layer(num_neurons, input_size, activation))
        else:
            # Busca la última capa que tenga neuronas (ignora Dropout para contar outputs)
            last_layer_size = 0
            for layer in reversed(self.layers):
                if hasattr(layer, 'n_neurons') and layer.n_neurons > 0:
                    last_layer_size = layer.n_neurons
                    break
            self.layers.append(Layer(num_neurons, last_layer_size, activation))
    
    def add_dropout(self, rate):
        self.layers.append(Dropout(rate))

    def forward(self, inputs, training=True):
        for layer in self.layers:
            inputs = layer.forward(inputs, training=training)
        return inputs
    
    def backward(self, loss_gradient):
        # Propagar el error hacia atrás
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)

    def train(self, x, y, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            # 1. Shuffle (Barajado)
            permutation = np.random.permutation(x.shape[0])
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]
            
            epoch_loss = 0
            
            # 2. Mini-Batch Gradient Descent
            for i in range(0, len(x), batch_size):
                # Crear batch
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                output = self.forward(x_batch, training=True)
                
                # Calcular Loss y Gradiente de Loss
                loss = self.loss_function.calculate(output, y_batch)
                epoch_loss += loss
                grad = self.loss_function.derivative(output, y_batch)
                
                # Backward (Calcula gradientes dW, db)
                self.backward(grad)
                
                # Optimizer Step (Actualiza pesos)
                self.optimizer.update(self.layers)
            
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss promedio: {epoch_loss / (len(x)/batch_size):.6f}")

    def predict(self, x):
        return self.forward(x, training=False)
    
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)