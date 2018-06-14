import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression


def get_data_1d():
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # iris virginica
    return X, y


def fit_logistic_regression(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr


def get_predictions_1d():
    X, y = get_data_1d()
    lr = fit_logistic_regression(X, y)
    X_test = [[.5], [1.0], [1.5], [1.7], [2.0], [2.5]]
    return lr.predict(X_test), lr.predict_proba(X_test)


def plot_predictions_1d():
    X, y = get_data_1d()
    lr = fit_logistic_regression(X, y)

    X_test = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = lr.predict_proba(X_test)
    not_virginica, virginica = y_proba[:, 0], y_proba[:, 1]

    decision_boundary = X_test[virginica >= 0.5][0]
    plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)

    plt.plot(X_test, virginica, 'g', label='virginica')
    plt.plot(X_test, not_virginica, 'b', label='not_virginica')
    plt.plot(X[y == 1], y[y == 1], 'gs')
    plt.plot(X[y == 0], y[y == 0], 'bs')
    plt.axis([0, 3, -0.1, 1.1])
    plt.legend()
    plt.title('Logistic regression with 1 feature')
    plt.xlabel('petal width')
    plt.ylabel('probability')
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    iris = datasets.load_iris()

    # predictions, probability = get_predictions_1d()
    # print(f'predictions: {predictions}\nprobability:\n{probability}')
    plot_predictions_1d()
