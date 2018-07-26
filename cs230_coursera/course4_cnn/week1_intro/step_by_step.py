import numpy as np
import matplotlib.pyplot as plt


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, pad_width=((0, 0), (pad, pad), (pad, pad), (0, 0)),
                   mode='constant', constant_values=(0, 0))
    return X_pad


if __name__ == '__main__':
    pass
