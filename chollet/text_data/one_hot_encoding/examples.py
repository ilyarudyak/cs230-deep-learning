import keras
import string
import numpy as np

from keras.preprocessing.text import Tokenizer


def get_tokens(samples):
    # First, build an index of all tokens in the data.
    token_index = {}
    for sample in samples:
        # We simply tokenize the samples via the `split` method.
        # in real life, we would also strip punctuation and special characters
        # from the samples.
        for word in sample.split():
            if word not in token_index:
                # Assign a unique index to each unique word
                token_index[word] = len(token_index) + 1
                # Note that we don't attribute index 0 to anything.
    return token_index


def vectorize(samples, token_index):
    # Next, we vectorize our samples.
    # We will only consider the first `max_length` words in each sample.
    max_length = 10
    # This is where we store our results:
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
    words = {}
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            words[(i, j)] = (word, index)
            results[i, j, index] = 1.0

    return words, results


def get_token_keras(samples):
    # We create a tokenizer, configured to only take
    # into account the top-1000 most common words
    tokenizer = Tokenizer(num_words=1000)
    # This builds the word index
    tokenizer.fit_on_texts(samples)

    # This turns strings into lists of integer indices.
    sequences = tokenizer.texts_to_sequences(samples)
    word_index = tokenizer.word_index

    return word_index, sequences, tokenizer


def vectorize_keras(tokenizer, samples):
    # You could also directly get the one-hot binary representations.
    # Note that other vectorization modes than one-hot encoding are supported!
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
    return one_hot_results


if __name__ == '__main__':
    # This is our initial data; one entry per "sample"
    # (in this toy example, a "sample" is just a sentence, but
    # it could be an entire document).
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']

    # token_index = get_tokens(samples)
    # words, results = vectorize(samples, token_index)
    # for k in words.keys():
    #     i, j = k
    #     word, index = words[k]
    #     print(f'{word:<10}:{index:<2}:{results[i, j, :]}')

    word_index, sequences, tokenizer = get_token_keras(samples)
    one_hot_results = vectorize_keras(tokenizer, samples)
    print(one_hot_results.shape)

