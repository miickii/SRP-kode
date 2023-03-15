import numpy as np
from layer import Layer
from helper_functions import one_hot

class NeuralNetwork:
    def __init__(self, layers=[]):
        self.layers = layers
        self.x_train = None
        self.y_train = None
        self.learning_rate = 0.001

    def add_layer(self, units, input_size, g, g_prime):
        self.layers.append(Layer(units, input_size, g, g_prime))

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        
        return output
    
    def backward(self, dA, alpha):
        for layer in reversed(self.layers):
            dA = layer.backward(dA, alpha)
    
    def get_accuracy(self, output, y):
        predictions = np.argmax(output, axis=0)
        return np.sum(predictions == y) / y.size
    
    def train(self, x_train, y_train, learning_rate, epochs):
        one_hot_y = one_hot(y_train)

        for i in range(epochs):
            output = self.forward(x_train)

            dA = output - one_hot_y
            self.backward(dA, learning_rate)

            if i % 1 == 0:
                print("Iteration: ", i)
                print("Accuracy: ", self.get_accuracy(output, y_train))
    
    def batch_loader(self, X, y=None, batch_size=100):
        m_samples = X.shape[1]
        for i in np.arange(0, m_samples, batch_size):
            begin, end = i, min(i + batch_size, m_samples)
            if y is not None:
                yield X.T[begin:end].T, y[begin: end]
            else:
                yield X.T[begin:end].T
    
    def batch_train(self, x_train, y_train, learning_rate, epochs):
        for i in range(epochs):
            accuracy = []
            for x_batch, y_batch in self.batch_loader(x_train, y_train):
                output = self.forward(x_batch)

                accuracy.append(self.get_accuracy(output, y_batch))

                dA = output - one_hot(y_batch)
                self.backward(dA, learning_rate)
            
            print("Iteration: ", i)
            print("Accuracy: ", np.mean(accuracy))
    
    def predict(self, X, y):
        output = self.forward(X)
        print("Accuracy: ", self.get_accuracy(output, y))

