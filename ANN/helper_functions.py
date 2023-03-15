import os
import numpy as np
import pandas as pd
import tensorflow as tf

def one_hot(y):
    # Først defineres en matrix
    # Antallet af rækker er antallet af træningseksempler
    # Antallet af søjler er mængden af kategorier (max() finder den højeste værdi, +1 fordi der tælles fra 0)
    one_hot_Y = np.zeros((y.size, y.max() + 1))

    # Nu går vi gennem alle rækker i one_hot_Y med np.arange(Y.size)
    # Ved hver række sættes elementet ved søjle Y[række] til 1
    # Hvis tredje Y værdi er 5 for eksempel, så er elementet ved tredje række og femte søjle i one_hot_Y 1,
    # og alle andre søjler i tredje række forbliver 0
    one_hot_Y[np.arange(y.size), y] = 1

    # Til sidst transponeres matricen så dens dimensioner passer med output matricen
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def load_doodle_data():
    x_train = np.empty((0, 784))
    y_train = np.empty((0,))
    x_test = np.empty((0, 784))
    y_test = np.empty((0,))
    for i, filename in enumerate(os.listdir("Data")):
        data = np.load(os.path.join("Data", filename))[0:10000]
        train_data = data[0:9000]
        test_data = data[9000:10000]
        
        x_train = np.concatenate([x_train, train_data])
        y_train = np.concatenate([y_train, np.ones(train_data.shape[0])*i])
        x_test = np.concatenate([x_test, test_data])
        y_test = np.concatenate([y_test, np.ones(test_data.shape[0])*i])
    
    perm = np.random.permutation(x_train.shape[0])
    # Shuffle both arrays in the same way
    x_train = x_train[perm]
    y_train = y_train[perm]
    
    return x_train.T, y_train.astype(int), x_test.T, y_test.astype(int)

def load_digit_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, -1))
    x_test = x_test.reshape((10000, -1))
    perm = np.random.permutation(x_train.shape[0])
    # Shuffle both arrays in the same way
    x_train = x_train[perm]
    y_train = y_train[perm]
    return x_train.T, y_train, x_test.T, y_test

def load_digit_data_from_csv():
    data = np.array(pd.read_csv('train.csv'))
    np.random.shuffle(data)
    m, n = data.shape

    data_test = data[0:1000].T
    y_test = data_test[0]
    x_test = data_test[1:n]

    data_train = data[1000:m].T
    y_train = data_train[0]
    x_train = data_train[1:n]
    return x_train, y_train, x_test, y_test

def sparse_categorical_crossentropy(y_true, y_pred, clip=True):

    y_true = tf.convert_to_tensor(y_true, dtype=tf.int32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    y_true = tf.one_hot(y_true, depth=y_pred.shape[1])

    if clip == True:
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    return - tf.reduce_mean(tf.math.log(y_pred[y_true == 1]))

def compare_SCCE(y_true, y_pred):
    print(tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred).numpy())
    print(sparse_categorical_crossentropy(y_true, y_pred, clip=True).numpy())
    print(sparse_categorical_crossentropy(y_true, y_pred, clip=False).numpy())

# y_true = [1, 2]
# y_pred = [[0.05, 0.95, 0.0], [0.1, 0.8, 0.1]]

# compare_SCCE(y_true, y_pred)
# 1.1769392
# 1.1769392
# 1.1769392

# y_true = [1, 2]
# y_pred = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]

# compare_SCCE(y_true, y_pred)
# 8.059048
# 8.059048
# inf