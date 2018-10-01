import numpy as np

from basic_rnn import rnn_cell_forward, rnn_forward


def test_rnn_cell_forward():
    np.random.seed(1)
    xt = np.random.randn(3, 10)
    a_prev = np.random.randn(5, 10)
    Waa = np.random.randn(5, 5)
    Wax = np.random.randn(5, 3)
    Wya = np.random.randn(2, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(2, 1)
    parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
    a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)

    target_a_next4 = np.array([0.59584544, 0.18141802, 0.61311866, 0.99808218, 0.85016201,
                               0.99980978, -0.18887155, 0.99815551, 0.6531151, 0.82872037])
    assert np.allclose(target_a_next4, a_next[4])

    assert a_next.shape == (5, 10)

    target_yt_pred1 = np.array([0.9888161, 0.01682021, 0.21140899, 0.36817467, 0.98988387,
                                0.88945212, 0.36920224, 0.9966312, 0.9982559, 0.17746526])
    assert np.allclose(target_yt_pred1, yt_pred[1])

    assert yt_pred.shape == (2, 10)


def test_rnn_forward():
    np.random.seed(1)
    x = np.random.randn(3, 10, 4)
    a0 = np.random.randn(5, 10)
    Waa = np.random.randn(5, 5)
    Wax = np.random.randn(5, 3)
    Wya = np.random.randn(2, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(2, 1)
    parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
    a, y_pred, caches = rnn_forward(x, a0, parameters)

    assert np.allclose(a[4][1], np.array([-0.99999375, 0.77911235, -0.99861469, -0.99833267]))
    assert a.shape == (5, 10, 4)
    assert np.allclose(y_pred[1][3], np.array([0.79560373, 0.86224861, 0.11118257, 0.81515947]))
    assert y_pred.shape == (2, 10, 4)
    assert np.allclose(caches[1][1][3], np.array([-1.1425182, -0.34934272, -0.20889423, 0.58662319]))
    assert len(caches) == 2
