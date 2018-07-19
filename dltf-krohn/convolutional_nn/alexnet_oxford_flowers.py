import numpy as np
import tflearn.datasets.oxflower17 as oxflower17

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard


def fit_alexnet(epochs=1, verbose=1):
    X, Y = oxflower17.load_data(one_hot=True)

    ############# build a model #############
    model = Sequential()

    model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(17, activation='softmax'))

    ############# build a model #############
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    tensorbrd = TensorBoard('logs/alexnet')
    model.fit(X, Y, batch_size=64, epochs=epochs, verbose=verbose, validation_split=0.1,
              shuffle=True, callbacks=[tensorbrd])


if __name__ == '__main__':
    np.random.seed(42)
    fit_alexnet(epochs=1, verbose=1)
