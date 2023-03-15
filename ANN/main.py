from activation_functions import ReLU, ReLU_prime, softmax, softmax_prime
from neural_network import NeuralNetwork
from helper_functions import load_digit_data, load_doodle_data

x_train, y_train, x_test, y_test = load_digit_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape, y_train.shape)

nn = NeuralNetwork()
nn.add_layer(512, 784, ReLU, ReLU_prime)
nn.add_layer(256, 512, ReLU, ReLU_prime)
nn.add_layer(10, 256, softmax, softmax_prime)

nn.train(x_train, y_train, 0.10, 5)
nn.predict(x_test, y_test)
# Digit:
# Accuracy med normal gradient descent: 0.8598
# Accuracy med batch gradient descent: 0.965

# Doodle:
# Accuracy med batch gradient descent: 0.8503