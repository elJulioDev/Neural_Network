import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.v_w = {} # Velocidad pesos
        self.v_b = {} # Velocidad bias

    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'weights'): # Solo capas con pesos (ignora Dropout)
                layer_id = id(layer)
                
                # Inicializar velocidades si es la primera vez
                if layer_id not in self.v_w:
                    self.v_w[layer_id] = np.zeros_like(layer.weights)
                    self.v_b[layer_id] = np.zeros_like(layer.biases)
                
                # Aplicar Momentum
                self.v_w[layer_id] = self.momentum * self.v_w[layer_id] - self.lr * layer.dweights
                self.v_b[layer_id] = self.momentum * self.v_b[layer_id] - self.lr * layer.dbiases
                
                # Actualizar parámetros
                layer.weights += self.v_w[layer_id]
                layer.biases += self.v_b[layer_id]