import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from fully_connected_nn.shallow_nn_mnist import preprocess_data


################################################################
# (1) using relu instead of sigmoid
# (2) add 1 more hidden layer
# (3) changed loss to categorical_crossentropy
# (4) changed learning rate to .1
# accuracy 0.9711 after 10 epochs
################################################################
def fit_shallow_nn(epochs=1):
    X_train, y_train, X_test, y_test = preprocess_data()
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(784,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=epochs,
              verbose=2, validation_data=(X_test, y_test))


if __name__ == '__main__':
    np.random.seed(42)
    fit_shallow_nn(10)
