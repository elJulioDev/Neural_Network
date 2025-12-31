import numpy as np

class Activation:
    def forward(self, x): pass
    def derivative(self, x): pass

class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    def derivative(self, x):
        s = self.forward(x)
        return s * (1 - s)

class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)
    def derivative(self, x):
        return np.where(x > 0, 1, 0)
    
class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def forward(self, x):
        return np.where(x > 0, x, x * self.alpha)
    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)

class Softmax(Activation):
    def forward(self, x):
        # Estabilidad numérica: restar el max
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    
    def derivative(self, x):
        # Truco: Usualmente Softmax se usa con CategoricalCrossEntropy
        # y la derivada simplificada de ambas juntas es (pred - y).
        # Por lo tanto, aquí retornamos 1 para que la Loss maneje el gradiente.
        return 1

class Linear(Activation):
    def forward(self, x):
        return x
    def derivative(self, x):
        return np.ones_like(x)