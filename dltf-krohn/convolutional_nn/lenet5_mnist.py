import numpy as np

import keras
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D


def preprocess_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # we now need spatial structure of our data
    X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
    X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')

    X_train /= 255
    X_test /= 255

    n_classes = 10
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    return X_train, y_train, X_test, y_test


def fit_lenet5(epochs=1):
    X_train, y_train, X_test, y_test = preprocess_data()
    n_classes = 10

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    tensorbrd = TensorBoard('logs/lenet5')
    model.fit(X_train, y_train, batch_size=128, epochs=epochs,
              verbose=1, validation_data=(X_test, y_test), callbacks=[tensorbrd])


if __name__ == '__main__':
    np.random.seed(42)
    fit_lenet5(epochs=1)
