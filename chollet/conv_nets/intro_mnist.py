from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import TensorBoard


def get_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model


def get_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


def train_model(model, train_images, train_labels, epochs=5):
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    callbacks = [TensorBoard(log_dir='logs',
                             histogram_freq=1,
                             embeddings_freq=1)]
    history = model.fit(x=train_images, y=train_labels,
                        epochs=epochs, batch_size=64,
                        validation_split=.2,
                        callbacks=callbacks)
    return history


if __name__ == '__main__':
    model = get_model()
    train_images, train_labels, test_images, test_labels = get_data()
    train_model(model, train_images, train_labels)

