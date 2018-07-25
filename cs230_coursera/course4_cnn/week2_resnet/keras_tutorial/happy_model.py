from keras import Input, Model
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, \
    Activation, MaxPooling2D, Flatten, Dense

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


def fit_happy_model(input_shape):
    hm = build_happy_model(input_shape)
    hm.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    hm.fit(X_train, Y_train, epochs=3, batch_size=64, verbose=2)
    hm.evaluate(x=X_test, y=Y_test)


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


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = preprocess_data()
    fit_happy_model(X_train.shape[1:])
