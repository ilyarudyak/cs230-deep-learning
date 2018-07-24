import os
import numpy as np
import matplotlib.pyplot as plt

from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/Users/ilyarudyak/data/cats_and_dogs_small'
npzfilename = 'vgg16_precomputed_weights.npz'


def extract_features(directory, sample_count, batch_size=20):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=sample_count)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= 40:  # sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


def precompute_data():
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    train_features, train_labels = extract_features(train_dir, 2000)
    validation_features, validation_labels = extract_features(validation_dir, 1000)
    test_features, test_labels = extract_features(test_dir, 1000)

    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

    np.savez(os.path.join(base_dir, npzfilename),
             train_features=train_features, train_labels=train_labels,
             validation_features=validation_features, validation_labels=validation_labels,
             test_features=test_features, test_labels=test_labels)


def get_precomputed_data():
    npzfile = np.load(os.path.join(base_dir, npzfilename))
    return npzfile['train_features'], npzfile['train_labels'], \
           npzfile['validation_features'], npzfile['validation_labels'], \
           npzfile['test_features'], npzfile['test_labels']


def fit_transfer_learning_model(epochs=1):
    train_features, train_labels, validation_features, validation_labels, _, _ = get_precomputed_data()

    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(train_features, train_labels, verbose=2,
                        epochs=epochs,
                        batch_size=20,
                        validation_data=(validation_features, validation_labels))
    return history


def plot_curves(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    # precompute_data()
    history = fit_transfer_learning_model(epochs=30)
    # plot_curves(history)
