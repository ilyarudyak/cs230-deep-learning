import numpy as np


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(x, axis=0)


def clip(gradients, max_value):
    for grad_value in gradients.values():
        # in-place clipping
        np.clip(grad_value, -max_value, max_value, out=grad_value)


def sample(parameters, char_to_ix, seed):
    pass