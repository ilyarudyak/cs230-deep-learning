import numpy as np

from blocks import clip, sample
from vocabulary import get_vocabulary


def test_clip():
    np.random.seed(3)
    dWax = np.random.randn(5, 3) * 10
    dWaa = np.random.randn(5, 5) * 10
    dWya = np.random.randn(2, 5) * 10
    db = np.random.randn(5, 1) * 10
    dby = np.random.randn(2, 1) * 10
    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya,
                 "db": db, "dby": dby}
    # in-place clipping
    clip(gradients, 10)

    assert gradients["dWaa"][1][2] == 10.0
    assert gradients["dWax"][3][1] == -10.0
    assert np.isclose(gradients["dWya"][1][2], 0.29713815361)
    assert gradients["db"][4] == [10.]
    assert np.allclose(gradients["dby"][1], np.array([8.45833407]))


def test_sample():
    vocab_to_ix, _ = get_vocabulary()
    vocab_size = len(vocab_to_ix)

    np.random.seed(2)
    n_a = 100
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), \
                    np.random.randn(vocab_size, n_a)
    ba, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": ba, "by": by}

    indices = sample(parameters, vocab_to_ix, 0)

    print(f'len(indices)={len(indices)}')
    print(indices)
