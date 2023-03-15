import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist

def plot_images(x, y):
    fig, ax = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            idx = random.randint(0, x.shape[0])
            ax[i, j].imshow(x[idx])
            ax[i, j].set_title(str(y[idx]))
    
    plt.show()

# Split the dataset into training and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.loadtxt('x_augmented.txt', dtype=int)
y_train = np.loadtxt('y_augmented.txt', dtype=int)

x_train = x_train.reshape((1140000, 28, 28))

#plot_images(x_train, y_train)

x_train, x_test = x_train / 255.0, x_test / 255.0

# Generate a random permutation of the indices
perm = np.random.permutation(x_train.shape[0])

# Shuffle both arrays in the same way
x_train = x_train[perm]
y_train = y_train[perm]

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test, verbose=2)

model.save('digit_model.h5')