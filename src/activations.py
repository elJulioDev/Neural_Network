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
        # Si x > 0 devuelve x, si no, devuelve x * 0.01
        return np.where(x > 0, x, x * self.alpha)
    
    def derivative(self, x):
        # Si x > 0 la derivada es 1, si no, es 0.01
        return np.where(x > 0, 1, self.alpha)