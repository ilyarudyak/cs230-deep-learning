import numpy as np


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


if __name__ == '__main__':
    # This is our initial data; one entry per "sample"
    # (in this toy example, a "sample" is just a sentence, but
    # it could be an entire document).
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    token_index = get_tokens(samples)
    words, results = vectorize(samples, token_index)

    for k in words.keys():
        i, j = k
        word, index = words[k]
        print(f'{word:<10}:{index:<2}:{results[i, j, :]}')

