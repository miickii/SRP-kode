import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate, shift

mnist = tf.keras.datasets.mnist

def rotate_image(img):
    return [img, rotate(img, 30, reshape=False), rotate(img, 330, reshape=False)]

def shift_image(img):
    return [img, shift(img, [3, 0]), shift(img, [-3, 0]), shift(img, [0, 3]), shift(img, [0, -3])]

def augment_image(img):
    rotated = rotate_image(img)
    shifted = [shift(img, [3, 3]), shift(img, [-3, -3]), shift(img, [3, -3]), shift(img, [-3, 3])]
    for i in rotated:
        shifted.extend(shift_image(i))
    
    return shifted

def augment_images(x, y):
    x_augmented = []
    y_augmented = []

    for i in range(x.shape[0]):
        augmented_imgs = augment_image(x[i])
        label = y[i]
        for img in augmented_imgs:
            x_augmented.append(img)
            y_augmented.append(label)

    return np.array(x_augmented), np.array(y_augmented)

# Split the datasæt i træningsdata og testdata
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_augmented, y_augmented = augment_images(x_train, y_train)
print(x_augmented.shape, y_augmented.shape)

flattened_images = x_augmented.reshape((1140000, -1))

np.savetxt('x_augmented.txt', flattened_images, fmt='%d')
np.savetxt('y_augmented.txt', y_augmented, fmt='%d')

#print(np.array_equal(x_augmented, x_train))
