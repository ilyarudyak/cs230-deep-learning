import numpy as np
import os

from sklearn.datasets import load_files
from keras.utils import np_utils


DATA_DIR = '/Users/ilyarudyak/data/dog_project/dogImages'
TRAIN, VALID, TEST = 'train', 'valid', 'test'
N_DOGS_BREED = 133


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), N_DOGS_BREED)
    return dog_files, dog_targets


def get_dog_files_short(length=10):
    train_files, train_targets = load_dataset(os.path.join(DATA_DIR, TRAIN))
    return train_files[:length]


def get_data():
    train_files, train_targets = load_dataset(os.path.join(DATA_DIR, TRAIN))
    valid_files, valid_targets = load_dataset(os.path.join(DATA_DIR, VALID))
    test_files, test_targets = load_dataset(os.path.join(DATA_DIR, TEST))
    return train_files, train_targets, valid_files, valid_targets, test_files, test_targets


if __name__ == '__main__':
    np.random.seed(42)
