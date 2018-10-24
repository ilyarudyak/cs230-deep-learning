from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Embedding, SimpleRNN, Dense
from keras.models import Sequential
from keras.layers import LSTM


def get_data():

    (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

    return input_train, y_train, input_test, y_test


def imdb_model_using_rnn(epochs=10):
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(input_train, y_train,
                        epochs=epochs,
                        batch_size=128,
                        validation_split=0.2)


def imdb_model_using_lstm(epochs=10):
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))  # the ONLY difference
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(input_train, y_train,
                        epochs=epochs,
                        batch_size=128,
                        validation_split=0.2,
                        verbose=2)


if __name__ == '__main__':
    max_features = 10000  # number of words to consider as features
    maxlen = 500  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    input_train, y_train, input_test, y_test = get_data()
    # imdb_model_using_rnn()
    imdb_model_using_lstm()