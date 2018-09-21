import numpy as np
import matplotlib.pyplot as plt


def get_data(input_images="/Users/ilyarudyak/data/gans/apple.npy"):
    data = np.load(input_images)
    data = data / 255
    data = np.reshape(data, (data.shape[0], 28, 28, 1))
    return data


if __name__ == '__main__':
    data = get_data()
    plt.imshow(data[4242, :, :, 0], cmap='Greys')
    plt.show()
