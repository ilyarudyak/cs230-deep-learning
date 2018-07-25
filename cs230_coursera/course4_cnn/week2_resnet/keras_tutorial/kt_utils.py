import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    # your train set features: (600, 64, 64, 3)
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    # your train set labels: (600,)
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    # your test set features:
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    # your test set labels: (150,)
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    # reshape them to row vectors (1, 600) and (1, 150)
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

