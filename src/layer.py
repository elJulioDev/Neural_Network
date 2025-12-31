import numpy as np

class Layer:
    def __init__(self, n_neurons, input_size, activation):
        self.n_neurons = n_neurons
        self.input_size = input_size
        self.activation = activation
        
        # Inicialización de He (Estándar profesional)
        # Weights: matriz (input_size x n_neurons)
        self.weights = np.random.randn(input_size, n_neurons) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, n_neurons))
        
        # Cache para backward pass
        self.inputs = None
        self.z = None
        self.dweights = None
        self.dbiases = None

    def forward(self, inputs, training=True):
        self.inputs = inputs
        # Operación matricial: (Batch x Input) dot (Input x Neurons) = (Batch x Neurons)
        self.z = np.dot(inputs, self.weights) + self.biases
        return self.activation.forward(self.z)

    def backward(self, d_output):
        # d_output viene de la capa siguiente: dL/dA
        # Multiplicamos por la derivada de la activación: dA/dZ
        d_activation = d_output * self.activation.derivative(self.z)
        
        # Calculamos gradientes
        m = self.inputs.shape[0] # Tamaño del batch
        
        # dW = X.T dot dZ
        self.dweights = np.dot(self.inputs.T, d_activation)
        
        # db = sum(dZ)
        self.dbiases = np.sum(d_activation, axis=0, keepdims=True)
        
        # dX (Error para la capa anterior) = dZ dot W.T
        d_input = np.dot(d_activation, self.weights.T)
        
        return d_input

class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate
        self.mask = None
        self.n_neurons = 0 # No tiene neuronas propias

    def forward(self, inputs, training=True):
        if training:
            # Máscara binomial (escala por 1/(1-rate) para mantener magnitud)
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
            return inputs * self.mask
        return inputs

    def backward(self, d_output):
        # El gradiente solo pasa por donde la máscara era 1
        return d_output * self.mask