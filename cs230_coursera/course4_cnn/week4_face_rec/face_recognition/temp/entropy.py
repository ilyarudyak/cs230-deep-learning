import numpy as np

# PXY = np.array([[1 / 8, 1 / 16, 1 / 32, 1 / 32],
#                 [1 / 16, 1 / 8, 1 / 32, 1 / 32],
#                 [1 / 16, 1 / 16, 1 / 16, 1 / 16],
#                 [1 / 4, 1e-10, 1e-10, 1e-10]
#                 ])

# PXY = np.array([[1e-10, 3/4],
#                 [1/8, 1/8]
#                 ])


def entropy(P):
    return np.round(cond_entropy(P, P), decimals=5)


def cond_entropy(P, PC):
    return -np.round(np.sum(P * np.log2(PC)), decimals=5)


def normalize(X):
    return X / np.sum(X)


PXY = normalize(np.array([[3, 6],
                          [4, 1]]))

PX = np.sum(PXY, axis=0)
PY = np.sum(PXY, axis=1)

PX_Y = PXY / PY.reshape(2, 1)
PY_X = PXY / PX.reshape(2, 1)


if __name__ == '__main__':
    pass
