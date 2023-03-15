import numpy as np

class Layer:
    def __init__(self, units, input_size, g, g_prime):
        self.input = None # Input til laget lagres til senere brug i backwardspropagation
        self.W = np.random.rand(units, input_size) - 0.5
        self.b = np.random.rand(units, 1) - 0.5
        self.Z = None # Z værdien lagres også efter den er beregnet, til brug i backwardspropagation
        self.g = g # Aktiveringsfunktion
        self.g_prime = g_prime # Aktiveringsfunktionen afledt
    
    # Tager aktiveringsværdierne fra forrige lag som input
    def forward(self, prev_A):
        self.input = prev_A

        # Sumfunktionen
        self.Z = np.dot(self.W, self.input) + self.b

        # Aktiveringsfunktionen
        A = self.g(self.Z)
        return A
    
    def backward(self, dA, learning_rate):
        dZ = dA * self.g_prime(self.Z)

        m = self.input.shape[1]
        dW = 1 / m * np.dot(dZ, self.input.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        prev_dA = np.dot(self.W.T, dZ)

        # Opdatering af parametre først efter gradienten af fejlfunktionen med hensyn til aktiveringsværdierne er beregnet
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return prev_dA