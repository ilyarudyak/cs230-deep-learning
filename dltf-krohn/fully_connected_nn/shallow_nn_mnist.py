import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


def get_raw_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def plot_digits():
    X_train, y_train, X_test, y_test = get_raw_data()
    size, figsize = 8, 6
    digits, labels = X_train[:size**2], y_train[:size**2]

    fig, axs = plt.subplots(8, size, figsize=(figsize, figsize))
    plt.subplots_adjust(hspace=.8)
    fig.suptitle('MNIST dataset: first {} images with labels'.format(size**2))
    for image, ax in enumerate(axs.flat):
        ax.imshow(digits[image], cmap='gray')
        ax.set(xticks=[], yticks=[])

        ax.set_title(str(labels[image]))
    plt.show()


def preprocess_data():
    X_train, y_train, X_test, y_test = get_raw_data()

    X_train = X_train.reshape(60000, 784).astype('float32')
    X_test = X_test.reshape(10000, 784).astype('float32')
    X_train /= 255
    X_test /= 255

    # one hot encoding
    n_classes = 10
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    return X_train, y_train, X_test, y_test


def fit_shallow_nn(epochs=1):
    X_train, y_train, X_test, y_test = preprocess_data()

    # There are two main types of models available in Keras:
    # the Sequential model, and the Model class used with the functional API.
    # The Sequential model is a linear stack of layers.
    # You can create a Sequential model by passing
    # a list of layer instances to the constructor.
    model = Sequential()

    # build the model
    model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, batch_size=128, epochs=epochs,
              verbose=2, validation_data=(X_test, y_test))


if __name__ == '__main__':
    np.random.seed(42)
    fit_shallow_nn(epochs=10)
