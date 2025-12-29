import numpy as np

class Neuron:

    def __init__(self, n_input):
        self.weights = np.random.randn(n_input)
        self.bias = np.random.randn()
        self.output = 0
        self.inputs = None
        self.dweight = np.zeros_like(self.weights)
        self.dbias = 0

    def activate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivate_activate(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        self.inputs = inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.output = self.activate(weighted_sum)
        return self.output
    
    def backward(self, d_output, learning_rate):
        d_activation = d_output * self.derivate_activate(self.output)
        self.dweight = np.dot(self.inputs, d_activation)
        self.dbias = d_activation
        d_input = np.dot(d_activation, self.weights)
        self.weights -= self.dweight * learning_rate
        self.bias -= learning_rate * self.dbias
        return d_input