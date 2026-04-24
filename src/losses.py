import numpy as np

class Loss:
    def calculate(self, output, y): pass
    def derivative(self, output, y): pass

class MSE(Loss):
    def calculate(self, output, y):
        return np.mean((y - output) ** 2)
    def derivative(self, output, y):
        return 2 * (output - y) / y.shape[0]

class BinaryCrossEntropy(Loss):
    def calculate(self, output, y):
        output = np.clip(output, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
    def derivative(self, output, y):
        output = np.clip(output, 1e-15, 1 - 1e-15)
        return (output - y) / (output * (1 - output)) / y.shape[0]

class CategoricalCrossEntropy(Loss):
    def calculate(self, output, y):
        # y debe ser one-hot encoded
        output = np.clip(output, 1e-15, 1 - 1e-15)
        return -np.sum(y * np.log(output)) / y.shape[0]
    
    def derivative(self, output, y):
        # Asumiendo combinación con Softmax
        return (output - y) / y.shape[0]