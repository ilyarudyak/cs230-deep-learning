from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense


def get_data_learn(max_features=10000, max_len=20):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    return x_train, y_train


def get_data_pretrained():
    x_train, y_train, x_val, y_val = 0, 0, 0, 0
    return x_train, y_train, x_val, y_val


def imdb_model_learn_embed(max_len=20):
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


def imdb_model_glove_embed(max_words, embedding_dim, max_len, embedding_matrix):

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))  # NEW layer
    model.add(Dense(1, activation='sigmoid'))

    # load glove embeddings in the model
    model.layers[0].set_weights([embedding_matrix])

    # freeze the embedding layer (NO training)
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val))


if __name__ == '__main__':
    # x_train, y_train = get_data_learn()
    # imdb_model_learn_embed()

    x_train, y_train, x_val, y_val = get_data_pretrained()