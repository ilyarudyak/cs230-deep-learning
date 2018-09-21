from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten


def discriminator_builder(depth=64, p=0.4):
    # Define inputs
    img_w, img_h = 28, 28
    inputs = Input((img_w, img_h, 1))

    # Convolutional layers
    conv1 = Conv2D(depth * 1, 5, strides=2, padding='same', activation='relu')(inputs)
    conv1 = Dropout(p)(conv1)

    conv2 = Conv2D(depth * 2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(depth * 4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)

    conv4 = Conv2D(depth * 8, 5, strides=1, padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(p)(conv4))

    # Output layer
    output = Dense(1, activation='sigmoid')(conv4)

    # Model definition
    model = Model(inputs=inputs, outputs=output)
    # model.summary()

    return model