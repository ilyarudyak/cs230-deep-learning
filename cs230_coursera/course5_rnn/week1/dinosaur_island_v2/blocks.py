import numpy as np


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(x, axis=0)


