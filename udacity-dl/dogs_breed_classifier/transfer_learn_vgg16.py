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

    VGG16_model = build_model(train_VGG16.shape[1:])

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5',
                                   verbose=1, save_best_only=True)

    VGG16_model.fit(train_VGG16, train_targets,
                    validation_data=(valid_VGG16, valid_targets),
                    epochs=20, batch_size=20, callbacks=[checkpointer], verbose=2)


def build_model(shape):
    VGG16_model = Sequential()
    VGG16_model.add(GlobalAveragePooling2D(input_shape=shape))
    VGG16_model.add(Dense(133, activation='softmax'))

    VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return VGG16_model


def test_model():
    bottleneck_features = np.load(os.path.join(BOTTLENECK_DIR, 'DogVGG16Data.npz'))
    train_VGG16 = bottleneck_features['train']
    test_VGG16 = bottleneck_features['test']

    _, _, _, _, _, test_targets = get_data()

    VGG16_model = build_model(train_VGG16.shape[1:])
    VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
    VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0)))
                         for feature in test_VGG16]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(VGG16_predictions) ==
                                 np.argmax(test_targets, axis=1)) / len(VGG16_predictions)
    print('test accuracy: {:.1f}%'.format(test_accuracy))


if __name__ == '__main__':
    # fit_model()
    test_model()
