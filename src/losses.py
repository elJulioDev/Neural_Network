import numpy as np

class Loss:
    def calculate(self, output, y): pass
    def derivative(self, output, y): pass

class MSE(Loss):
    def calculate(self, output, y):
        return np.mean((y - output) ** 2)
    
    def derivative(self, output, y):
        return 2 * (output - y) / y.size # Normalizado por tamaño

class BinaryCrossEntropy(Loss):
    def calculate(self, output, y):
        # Clip para evitar log(0)
        output = np.clip(output, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))

    def derivative(self, output, y):
        output = np.clip(output, 1e-15, 1 - 1e-15)
        return (output - y) / (output * (1 - output)) / y.size