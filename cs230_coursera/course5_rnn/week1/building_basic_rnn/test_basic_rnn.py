import numpy as np

from building_basic_rnn.basic_rnn import rnn_cell_forward, rnn_forward, lstm_cell_forward, lstm_forward


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


def test_lstm_cell_forward():
    np.random.seed(1)
    xt = np.random.randn(3, 10)
    a_prev = np.random.randn(5, 10)
    c_prev = np.random.randn(5, 10)
    Wf = np.random.randn(5, 5 + 3)
    bf = np.random.randn(5, 1)
    Wi = np.random.randn(5, 5 + 3)
    bi = np.random.randn(5, 1)
    Wo = np.random.randn(5, 5 + 3)
    bo = np.random.randn(5, 1)
    Wc = np.random.randn(5, 5 + 3)
    bc = np.random.randn(5, 1)
    Wy = np.random.randn(2, 5)
    by = np.random.randn(2, 1)
    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf,
                  "bi": bi, "bo": bo, "bc": bc, "by": by}
    a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

    assert np.allclose(a_next[4], np.array([-0.66408471, 0.0036921, 0.02088357, 0.22834167, -0.85575339,
                                            0.00138482, 0.76566531, 0.34631421, -0.00215674, 0.43827275]))
    assert a_next.shape == (5, 10)
    assert np.allclose(c_next[2], np.array([0.63267805, 1.00570849, 0.35504474, 0.20690913, -1.64566718,
                                            0.11832942, 0.76449811, -0.0981561, -0.74348425, -0.26810932]))
    assert c_next.shape == (5, 10)
    assert np.allclose(yt[1], np.array([0.79913913, 0.15986619, 0.22412122, 0.15606108, 0.97057211,
                                        0.31146381, 0.00943007, 0.12666353, 0.39380172, 0.07828381]))
    assert yt.shape == (2, 10)
    assert np.allclose(cache[1][3], np.array([-0.16263996, 1.03729328, 0.72938082, -0.54101719, 0.02752074,
                                              -0.30821874, 0.07651101, -1.03752894, 1.41219977, -0.37647422]))
    assert len(cache) == 10


def test_lstm_forward():
    np.random.seed(1)
    x = np.random.randn(3, 10, 7)
    a0 = np.random.randn(5, 10)
    Wf = np.random.randn(5, 5 + 3)
    bf = np.random.randn(5, 1)
    Wi = np.random.randn(5, 5 + 3)
    bi = np.random.randn(5, 1)
    Wo = np.random.randn(5, 5 + 3)
    bo = np.random.randn(5, 1)
    Wc = np.random.randn(5, 5 + 3)
    bc = np.random.randn(5, 1)
    Wy = np.random.randn(2, 5)
    by = np.random.randn(2, 1)
    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf,
                  "bi": bi, "bo": bo, "bc": bc, "by": by}
    a, y, c, caches = lstm_forward(x, a0, parameters)

    assert np.isclose(a[4][3][6], 0.73162451027)
    assert a.shape == (5, 10, 7)
    assert np.isclose(y[1][4][3], 0.95087346185)
    assert y.shape == (2, 10, 7)
    assert np.allclose(caches[1][1][1], np.array([0.82797464, 0.23009474, 0.76201118, -0.22232814,
                                                  -0.20075807, 0.18656139, 0.41005165]))
    assert np.isclose(c[1][2][1], -0.855544916718)
    assert len(caches) == 2
