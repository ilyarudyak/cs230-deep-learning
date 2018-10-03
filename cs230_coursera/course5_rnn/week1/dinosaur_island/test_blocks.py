import numpy as np

from blocks import clip


def test_clip():
    np.random.seed(3)
    dWax = np.random.randn(5, 3) * 10
    dWaa = np.random.randn(5, 5) * 10
    dWya = np.random.randn(2, 5) * 10
    db = np.random.randn(5, 1) * 10
    dby = np.random.randn(2, 1) * 10
    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya,
                 "db": db, "dby": dby}
    gradients = clip(gradients, 10)

    assert gradients["dWaa"][1][2] == 10.0
    assert gradients["dWax"][3][1] == -10.0
    assert np.isclose(gradients["dWya"][1][2], 0.29713815361)
    assert gradients["db"][4] == [10.]
    assert np.allclose(gradients["dby"][1], np.array([8.45833407]))

