import numpy as np
import matplotlib.pyplot as plt

from step_by_step import zero_pad


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


if __name__ == '__main__':
    test_zero_pad()
