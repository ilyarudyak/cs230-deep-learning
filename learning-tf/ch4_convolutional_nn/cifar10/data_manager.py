import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "/Users/ilyarudyak/data/cifar-10-batches-py"


def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as f:
        return pickle.load(f, encoding='latin1')


def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()


def get_images_by_class(images, labels, ncol, nclasses=10):
    images_by_class = []
    for i in range(nclasses):
        mask = labels[:, i] == 1
        images_by_class.append(images[mask][:ncol])
    return images_by_class


def display_cifar_by_class(images, labels, label_names, nrow=10, ncol=10, size=6):
    images_by_class = get_images_by_class(images, labels, ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(size, size))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images_by_class[i // ncol][i % ncol], cmap='binary')
        ax.set(xticks=[], yticks=[])
        if i % ncol == 0:
            ax.set_ylabel(label_names[i // ncol], rotation='horizontal',
                          ha='right', fontsize='medium', fontweight='bold')
    plt.show()


class CifarLoader(object):
    """
    Load and manage the CIFAR dataset.
    (for any practical use there is no reason not to use the built-in dataset handler instead)
    """

    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None
        self.num_examples = 0


    def load(self):
        # unpickle all files in source list into list of dict
        # dict_keys(['batch_label', 'labels', 'data', 'filenames'])
        # data['labels'][:10] >> [6, 9, 9, 4, 1, 1, 2, 7, 8, 3]
        # data['data'].shape >> (10000, 3072) ## 32 * 32 * 3 = 3072
        data = [unpickle(f) for f in self._source]

        # get images from all dictionaries and stack them vertically
        # images.shape >> (50000, 3072) ## for train data
        images = np.vstack([d['data'] for d in data])
        n = len(images)  # the same as images.shape[0]

        # we need spatial structure for CNN so we reshaped back to 3D images
        # and normalize
        # why do we reshape and transpose like this?
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        self.num_examples = self.images.shape[0]

        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i + batch_size], \
               self.labels[self._i:self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

    def random_batch(self, batch_size):
        n = len(self.images)
        ix = np.random.choice(n, batch_size)
        return self.images[ix], self.labels[ix]


class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)]).load()
        self.test = CifarLoader(["test_batch"]).load()
        label_names_dict = unpickle("batches.meta")
        self.label_names = label_names_dict['label_names']
        self.nclasses = 10

    def display_by_class(self, nrow=10, ncol=10, size=6):
        images_by_class = self.get_images_by_class()
        fig, axs = plt.subplots(nrow, ncol, figsize=(size, size))
        for i, ax in enumerate(axs.flat):
            ax.imshow(images_by_class[i // ncol][i % ncol], cmap='binary')
            ax.set(xticks=[], yticks=[])
            if i % ncol == 0:
                ax.set_ylabel(self.label_names[i // ncol], rotation='horizontal',
                              ha='right', fontsize='medium', fontweight='bold')
        plt.show()

    def get_images_by_class(self):
        images_by_class = []
        for i in range(self.nclasses):
            mask = self.train.labels[:, i] == 1
            images_by_class.append(self.train.images[mask][:self.nclasses])
        return images_by_class


def create_cifar_image():
    cdm = CifarDataManager()
    print("Number of train images: {}".format(len(cdm.train.images)))
    print("Number of train labels: {}".format(len(cdm.train.labels)))
    print("Number of test images: {}".format(len(cdm.test.images)))
    print("Number of test labels: {}".format(len(cdm.test.labels)))
    images = cdm.train.images
    display_cifar(images, 10)


if __name__ == "__main__":
    cifar10 = CifarDataManager()
    cifar10.display_by_class()
