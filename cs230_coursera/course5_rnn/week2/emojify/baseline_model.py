import emoji
import numpy as np
import matplotlib.pyplot as plt

from emo_utils import read_csv, convert_to_one_hot, read_glove_vecs, \
    predict, softmax, print_predictions


def get_data():
    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/tesss.csv')
    Y_oh_train = convert_to_one_hot(Y_train, C=5)
    Y_oh_test = convert_to_one_hot(Y_test, C=5)
    maxLen = len(max(X_train, key=len).split())
    word_to_index, index_to_word, word_to_vec_map = \
        read_glove_vecs('../word_vectors/data/glove.6B.50d.txt')
    return X_train, Y_train, X_test, Y_test, word_to_vec_map


def sentence_to_avg(sentence, word_to_vec_map):
    sv = np.array([word_to_vec_map[word] for word in sentence.lower().split()])
    return np.average(sv, axis=0)


def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    Model to train word vector representations in numpy.

    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 4, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its
    50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations

    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    np.random.seed(1)

    # Define number of training examples
    m, n_y, n_h = X.shape[0], 5, 50

    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, n_y)

    pred, cost = 0, 0
    for t in range(num_iterations):
        for i in range(m):

            # average word vectors
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # forward pass
            z = W.dot(avg) + b
            a = softmax(z)

            # compute cost
            cost = -np.sum(Y_oh[i] * np.log(a))

            # backward pass
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            # update params
            W -= learning_rate * dW
            b -= learning_rate * db

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, word_to_vec_map = get_data()
    pred, W, b = model(X_train, Y_train, word_to_vec_map, num_iterations=400)

    # pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
    # pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

    X_my_sentences = np.array(
        ["i adore you", "i love you", "funny lol", "lets play with a ball",
         "food is ready", "not feeling happy"])
    Y_my_labels = np.array([[0], [0], [2], [1], [4], [3]])

    pred = predict(X_my_sentences, Y_my_labels, W, b, word_to_vec_map)
    print_predictions(X_my_sentences, pred)
