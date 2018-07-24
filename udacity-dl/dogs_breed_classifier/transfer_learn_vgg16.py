import numpy as np
import os

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dense

from data_manager import get_data

BOTTLENECK_DIR = '/Users/ilyarudyak/data/dog_project/'


def fit_model():
    bottleneck_features = np.load(os.path.join(BOTTLENECK_DIR, 'DogVGG16Data.npz'))
    train_VGG16 = bottleneck_features['train']
    valid_VGG16 = bottleneck_features['valid']

    _, train_targets, _, valid_targets, _, _ = get_data()

    VGG16_model = Sequential()
    VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
    VGG16_model.add(Dense(133, activation='softmax'))

    VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5',
                                   verbose=1, save_best_only=True)

    VGG16_model.fit(train_VGG16, train_targets,
                    validation_data=(valid_VGG16, valid_targets),
                    epochs=20, batch_size=20, callbacks=[checkpointer], verbose=2)


if __name__ == '__main__':
    fit_model()
