import numpy as np


from blocks import rnn_step_forward, rnn_step_backward, \
    update_parameters, clip, rnn_forward, rnn_backward


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
