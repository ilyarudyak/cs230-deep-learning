from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense


def get_data(max_features=10000, max_len=20):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    return x_train, y_train


def imdb_model_keras(max_len=20):
    model = Sequential()

    # We specify the maximum input length to our Embedding layer
    # so we can later flatten the embedded inputs
    model.add(Embedding(10000, 8, input_length=max_len))

    # After the Embedding layer,
    # our activations have shape `(samples, max_len, 8)`.

    # We flatten the 3D tensor of embeddings
    # into a 2D tensor of shape `(samples, max_len * 8)`
    model.add(Flatten())

    # We add the classifier on top
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2)


if __name__ == '__main__':
    x_train, y_train = get_data()
    imdb_model_keras()
