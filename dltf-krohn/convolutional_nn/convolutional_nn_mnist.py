import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout # new!
from keras.layers.normalization import BatchNormalization # new!
from keras import regularizers # new!
from keras.optimizers import SGD


def preprocess_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784).astype('float32')
    X_test = X_test.reshape(10000, 784).astype('float32')

    X_train /= 255
    X_test /= 255

    n_classes = 10
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    return X_train, y_train, X_test, y_test


def fit_convolutional_nn(epochs=1):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(784,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=epochs,
              verbose=2, validation_data=(X_test, y_test))


if __name__ == '__main__':
    np.random.seed(42)
    X_train, y_train, X_test, y_test = preprocess_data()
    fit_convolutional_nn(epochs=1)
