import numpy as np

class Neuron:
    def __init__(self, n_input, activation):
        # Ayuda a que las neuronas ReLU no nazcan muertas
        # Multiplicamos por la raíz de (2 / n_input) para mantener la varianza
        self.weights = np.random.randn(n_input) * np.sqrt(2 / n_input)
        
        # El sesgo (bias) debe iniciar en 0 para no empujar a negativo desde el principio
        self.bias = 0.0
        # Inicializamos variables
        self.weighted_sum = 0
        self.output = 0
        self.inputs = None
        self.dweight = np.zeros_like(self.weights)
        self.dbias = 0
        self.activation = activation

    def activate(self, x):
        return self.activation.forward(x)
    
    def derivate_activate(self, x):
        return self.activation.derivative(x)
    
    def forward(self, inputs):
        self.inputs = inputs
        # Guardamos Z (suma ponderada) antes de activar
        self.weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Activamos Z
        self.output = self.activate(self.weighted_sum)
        return self.output
    
    def backward(self, d_output, learning_rate):
        # CORRECCIÓN CRÍTICA:
        # Usamos self.weighted_sum para calcular la derivada, NO self.output
        d_activation = d_output * self.derivate_activate(self.weighted_sum)
        
        self.dweight = np.dot(self.inputs, d_activation)
        self.dbias = d_activation
        d_input = np.dot(d_activation, self.weights)
        
        self.weights -= self.dweight * learning_rate
        self.bias -= learning_rate * self.dbias
        return d_input