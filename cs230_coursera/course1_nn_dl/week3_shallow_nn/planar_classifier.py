import numpy as np
import matplotlib.pyplot as plt
# from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from course1_nn_dl.week3_shallow_nn.planar_utils import plot_decision_boundary, \
    sigmoid, load_planar_dataset, load_extra_datasets


def plot_planar():
    plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
    plt.show()


def fit_logistic_regression():
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)

    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    X, Y = load_planar_dataset()

    # plot_planar()
    fit_logistic_regression()

