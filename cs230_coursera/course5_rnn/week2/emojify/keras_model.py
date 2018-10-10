import numpy as np

np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)

from emo_utils import read_csv, convert_to_one_hot, read_glove_vecs


def get_data():
    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/tesss.csv')
    Y_oh_train = convert_to_one_hot(Y_train, C=5)
    Y_oh_test = convert_to_one_hot(Y_test, C=5)
    maxLen = len(max(X_train, key=len).split())
    word_to_index, index_to_word, word_to_vec_map = \
        read_glove_vecs('../word_vectors/data/glove.6B.50d.txt')
    return word_to_index, index_to_word, word_to_vec_map


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]
    X_indices = np.zeros((m, max_len))

    for i in range(m):
        l, indicies = get_indices(X[i], word_to_index)
        X_indices[i, 0:l] = indicies

    return X_indices


def get_indices(sentence, word_to_index):
    words = sentence.split()
    return len(words), np.array([word_to_index[word] for word in words])


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1
    emb_dim = 50

    # Initialize the embedding matrix as a numpy array of zeros of shape
    # (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector
    # representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it non-trainable.
    # Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


if __name__ == '__main__':
    word_to_index, index_to_word, word_to_vec_map = get_data()

    # X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
    # X1_indices = sentences_to_indices(X1, word_to_index, max_len=5)
    # print("X1 =", X1)
    # print("X1_indices =", X1_indices)

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
