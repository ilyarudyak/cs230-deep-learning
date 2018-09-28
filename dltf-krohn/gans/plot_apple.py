import numpy as np
from matplotlib import pyplot as plt


def plotting_apple():
    plt.figure(figsize=(5, 5))

    for k in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, k + 1)
        plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    gen_imgs = np.load('gen_img1000.npy')
    plotting_apple()
