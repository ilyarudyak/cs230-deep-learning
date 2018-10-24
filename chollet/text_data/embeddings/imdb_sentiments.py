import os
import numpy as np

from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def get_data_learn(max_features=10000, max_len=20):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    return x_train, y_train


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


################### GloVe embedding ###################


def get_raw_reviews():
    imdb_dir = '/Users/ilyarudyak/data/imdb/aclImdb'
    train_dir = os.path.join(imdb_dir, 'train')

    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        print(f'starting {label_type} ...')
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    return texts, labels


def get_preprocessed_reviews(texts, labels):
    maxlen = 100  # We will cut reviews after 100 words
    training_samples = 200  # We will be training on 200 samples
    validation_samples = 10000  # We will be validating on 10000 samples
    max_words = 10000  # We will only consider the top 10,000 words in the dataset

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=maxlen)

    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # Split the data into a training set and a validation set
    # But first, shuffle the data, since we started from data
    # where sample are ordered (all negative first, then all positive).
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    return x_train, y_train, x_val, y_val, word_index


def get_glove_embeddings():
    glove_dir = '/Users/ilyarudyak/data/glove'

    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def get_embedding_matrix(word_index, max_words=10000):
    embedding_dim = 50

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def imdb_model_glove_embed(embedding_matrix, max_words=10000, embedding_dim=50, maxlen=100):

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
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
    model.save_weights('pre_trained_glove_model.h5')


if __name__ == '__main__':
    # x_train, y_train = get_data_learn()
    # imdb_model_learn_embed()

    texts, labels = get_raw_reviews()
    x_train, y_train, x_val, y_val, word_index = get_preprocessed_reviews(texts, labels)
    embeddings_index = get_glove_embeddings()
    embedding_matrix = get_embedding_matrix(word_index)
    imdb_model_glove_embed(embedding_matrix)
