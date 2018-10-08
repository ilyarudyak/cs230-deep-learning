import numpy as np


from blocks import rnn_step_forward, rnn_step_backward, \
    update_parameters, clip, rnn_forward, rnn_backward, \
    initialize_parameters, get_initial_loss, smooth, \
    sample, print_sample


from vocabulary import get_vocabulary


def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    """
    One step of gradient descent: forward, backward and parameter update.
    """
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    clip(gradients, 5)
    # update parameters in place
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X)-1]


def model(data, ix_to_vocab, vocab_to_ix, num_iterations=5000, n_a=50,
          dino_names=7, vocab_size=27):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text, size of the vocabulary

    Returns:
    parameters -- learned parameters
    """

    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss, don't worry about it)
    loss = get_initial_loss(vocab_size, dino_names)

    # Build list of all dinosaur names (training examples).
    # examples = ['aachenosaurus', 'aardonyx', 'abdallahsaurus'...]
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(num_iterations):

        ### START CODE HERE ###

        # Use the hint above to define one training example (X,Y) (≈ 2 lines)
        index = j % len(examples)
        X = [None] + [vocab_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [vocab_to_ix["\n"]]

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.01)

        ### END CODE HERE ###

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                # Sample indices and print them
                sampled_indices = sample(parameters, vocab_to_ix, seed)
                print_sample(sampled_indices, ix_to_vocab)

                seed += 1  # To get the same result for grading purposed, increment the seed by one.

            print('\n')

    return parameters


if __name__ == '__main__':
    dinos_str, vocab_to_ix, ix_to_vocab = get_vocabulary()
    parameters = model(dinos_str, ix_to_vocab, vocab_to_ix)
