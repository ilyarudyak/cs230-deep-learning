import numpy as np
import matplotlib.pyplot as plt

from step_by_step import zero_pad, conv_single_step, conv_forward


def test_zero_pad():
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 2)

    assert x.shape == (4, 3, 3, 2)
    assert x_pad.shape == (4, 7, 7, 2)

    target_array = np.array([[0.90085595, -0.68372786], [-0.12289023, -0.93576943], [-0.26788808, 0.53035547]])
    assert np.allclose(x[1, 1], target_array)

    target_array2 = np.array([[0., 0.],
                              [0., 0.],
                              [0., 0.],
                              [0., 0.],
                              [0., 0.],
                              [0., 0.],
                              [0., 0.]])
    assert np.array_equal(x_pad[1, 1], target_array2)


def test_conv_single_step():
    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)

    Z = conv_single_step(a_slice_prev, W, b)
    print(type(Z))
    assert Z == -6.999089450680221


def test_conv_forward():
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2,
                   "stride": 2}

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    assert np.isclose(np.mean(Z), 0.0489952035289)
    assert np.allclose(Z[3, 2, 1], np.array([-0.61490741, -6.7439236,
                                             -2.55153897, 1.75698377,
                                             3.56208902, 0.53036437,
                                             5.18531798, 8.75898442]))
    assert np.allclose(cache_conv[0][1][2][3], np.array([-0.20075807, 0.18656139, 0.41005165]))


if __name__ == '__main__':
    test_zero_pad()
