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

    fig, axs = plt.subplots(1, size, figsize=(figsize, figsize))
    fig.suptitle(f'MNIST dataset: first {size**2} images')
    for image, ax in enumerate(axs.flat):
        ax.imshow(digits[image], cmap='gray')
        ax.set(xticks=[], yticks=[])
        ax.set_title(str(labels[image]))
    plt.show()


def preprocess_data():
    pass


if __name__ == '__main__':
    np.random.seed(42)
    plot_digits()
