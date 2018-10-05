import numpy as np


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(x, axis=0)


def clip(gradients, max_value):
    for grad_value in gradients.values():
        # in-place clipping
        np.clip(grad_value, -max_value, max_value, out=grad_value)


def sample(parameters, vocab_to_ix, seed):
    """
    So the main goal of this function is to generate some sequence of characters
    (again this is a character-by-character model). We start from x1 == 0 and a0 == 0.
    We use weights W, b that we've learnt. So we propagate through one RNN unit and get y1.
    We use it as a probability distribution to sample a character. Then we use this sampled character
    as x2 - input to the 2nd RNN unit.
    """

    # extract parameters
    Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], \
                            parameters['by'], parameters['b']  # here we have 'b', not 'ba'
    vocab_size = len(vocab_to_ix)
    n_a, _ = Waa.shape

    ### START CODE HERE ###
    # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation).
    # that is a character-by-character model, so x - is one-hot vector with len == vocab_size
    # (our vocabulary is just an alphabet + '\n')
    x = np.zeros((vocab_size, 1))
    # Step 1': Initialize a_prev as zeros (≈1 line)
    a_prev = np.zeros((n_a, 1))

    # Create an empty list of indices, this is the list which will contain the list of indices
    # of the characters that we sampled.
    indices = []

    # Idx is a flag to detect a newline character, we initialize it to -1
    idx = -1

    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append
    # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well
    # trained model), which helps debugging and prevents entering an infinite loop.
    counter = 0
    newline_character_idx = vocab_to_ix['\n']

    while idx != newline_character_idx and counter != 50:
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(Waa.dot(a_prev) + Wax.dot(x) + by)
        z = Wya.dot(a) + ba
        y = softmax(z)

        # for grading purposes
        np.random.seed(counter + seed)

        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(np.arange(vocab_size), p=y.ravel())

        # Append the index to "indices"
        indices.append(idx)

        # Step 4: Overwrite the input character as the one corresponding to the sampled index.
        x = np.zeros((vocab_size, 1))  # so we use generated index as an input to the next RNN block
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

        # for grading purposes
        seed += 1
        counter += 1

    ### END CODE HERE ###

    if counter == 50:
        indices.append(vocab_to_ix['\n'])

    return indices
