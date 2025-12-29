import numpy as np
from .neuron import Neuron

class Layer:
    def __init__(self, num_neurons, inputs_size):
        self.neurons = [Neuron(inputs_size) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])
    
    def backward(self, d_outputs, learning_rate):
        d_inputs = np.zeros_like(self.neurons[0].inputs)
        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_outputs[i], learning_rate)
        return d_inputs