import numpy as np
import tensorflow as tf

from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, \
    Activation, MaxPooling2D, Flatten, Dense, Dropout

from kt_utils import load_dataset


def preprocess_data():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, \
    classes = load_dataset()

    # normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    return X_train, Y_train, X_test, Y_test


def fit_happy_model(input_shape, epochs=1):
    hm = build_happy_model2(input_shape)
    hm.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

    callbacks_list = [
        EarlyStopping(monitor='val_loss',
                      patience=10),
        ModelCheckpoint(filepath=file_path,
                        monitor='val_loss',
                        verbose=1, save_best_only=True),
        TensorBoard('logs/happy_house')
    ]
    hm.fit(X_train, Y_train,
           epochs=epochs, batch_size=64, verbose=2,
           validation_split=0.1,
           callbacks=callbacks_list)

    return hm


def build_happy_model2(input_shape):

    X_input = Input(input_shape)

    # CONV->BATCHNORM->RELU
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), name='conv0')(X_input)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), name='max_pool0')(X)

    # CONV->BATCHNORM->RELU
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    # FLATTEN->DENSE->OUTPUT
    X = Flatten()(X)
    X = Dense(64, activation='relu', name='dense0')(X)
    X = Dropout(rate=.5)(X)
    X_out = Dense(1, activation='sigmoid', name='out0')(X)

    model = Model(inputs=X_input, outputs=X_out, name='happy_model')

    return model


def build_happy_model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape.
    # Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll
    # use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='HappyModel')
    return model


def evaluate_model(hm):
    hm.load_weights(file_path)
    test_loss, test_accuracy = hm.evaluate(x=X_test, y=Y_test, verbose=0)
    return test_loss, test_accuracy


if __name__ == '__main__':
    np.random.seed(42)
    file_path = 'saved_models/best_weights.h5'

    X_train, Y_train, X_test, Y_test = preprocess_data()
    hm = fit_happy_model(X_train.shape[1:], epochs=3)
    # print(hm.summary())
    test_loss, test_accuracy = evaluate_model(hm)
    print(f'test_loss:{test_loss} test_accuracy:{test_accuracy}')
