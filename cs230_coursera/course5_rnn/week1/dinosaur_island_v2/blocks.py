import numpy as np


######################################################
#################### misc functions ##################
######################################################


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex, axis=0)


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
        a = np.tanh(Waa.dot(a_prev) + Wax.dot(x) + ba)
        z = Wya.dot(a) + by
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


def print_sample(sample_ix, ix_to_vocab):
    txt = ''.join(ix_to_vocab[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    print('%s' % (txt,), end='')


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length


def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values

    Returns:
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x) * 0.01  # input to hidden
    Waa = np.random.randn(n_a, n_a) * 0.01  # hidden to hidden
    Wya = np.random.randn(n_y, n_a) * 0.01  # hidden to output
    b = np.zeros((n_a, 1))  # hidden bias
    by = np.zeros((n_y, 1))  # output bias

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

    return parameters


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


######################################################
#################### single steps ####################
######################################################


def rnn_step_forward(parameters, a_prev, x):
    Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], \
                            parameters['by'], parameters['b']
    a_next = np.tanh(Waa.dot(a_prev) + Wax.dot(x) + ba)
    y_t = softmax(Wya.dot(a_next) + by)
    return a_next, y_t


def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next']  # backprop into h
    daraw = (1 - a * a) * da  # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients


def update_parameters(parameters, gradients, lr):
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b'] += -lr * gradients['db']
    parameters['by'] += -lr * gradients['dby']
    return parameters


######################################################
#################### full pass    ####################
######################################################


def rnn_forward(X, Y, a0, parameters, vocab_size=27):
    # Initialize x, a and y_hat as empty dictionaries
    x, a, y_hat = {}, {}, {}

    a[-1] = np.copy(a0)

    # initialize your loss to 0
    loss = 0

    for t in range(len(X)):

        # Set x[t] to be the one-hot vector representation of the t'th character in X.
        # if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector.
        x[t] = np.zeros((vocab_size, 1))
        if X[t] is not None:
            x[t][X[t]] = 1

        # Run one step forward of the RNN
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t - 1], x[t])

        # Update the loss by substracting the cross-entropy term of this time-step from it.
        loss -= np.log(y_hat[t][Y[t], 0])

    cache = (y_hat, a, x)

    return loss, cache


def rnn_backward(X, Y, parameters, cache):
    # Initialize gradients as an empty dictionary
    gradients = {}

    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])

    ### START CODE HERE ###
    # Backpropagate through time
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t - 1])
    ### END CODE HERE ###

    return gradients, a
