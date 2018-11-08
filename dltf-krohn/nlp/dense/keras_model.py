import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt


def get_index():
    word_index = keras.datasets.imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["PAD"] = 0
    word_index["START"] = 1
    word_index["UNK"] = 2
    index_word = {v: k for k, v in word_index.items()}

    return word_index, index_word


def get_data(n_unique_words=5000, n_words_to_skip=50, max_review_length=100,
             pad_type='pre', trunc_type='pre', pad_value=0):
    (x_train, y_train), (x_valid, y_valid) = \
        imdb.load_data(num_words=n_unique_words, skip_top=n_words_to_skip)

    x_train = pad_sequences(x_train, maxlen=max_review_length,
                            padding=pad_type, truncating=trunc_type, value=pad_value)
    x_valid = pad_sequences(x_valid, maxlen=max_review_length,
                            padding=pad_type, truncating=trunc_type, value=pad_value)

    return x_train, y_train, x_valid, y_valid


def fit_model(n_unique_words=5000, max_review_length=100,
              n_dim=64, n_dense=64, dropout=0.5,
              epochs=4, batch_size=128):
    model = Sequential()
    model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
    model.add(Flatten())
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dropout(dropout))
    # model.add(Dense(n_dense, activation='relu'))
    # model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
              validation_data=(x_valid, y_valid))


def get_review_text(index):
    return ' '.join(index_word[id] for id in all_x_train[index])


if __name__ == '__main__':
    # output directory name:
    output_dir = 'model_output'

    x_train, y_train, x_valid, y_valid = get_data()
    (all_x_train, _), (all_x_valid, _) = imdb.load_data()
    word_index, index_word = get_index()
    fit_model()
