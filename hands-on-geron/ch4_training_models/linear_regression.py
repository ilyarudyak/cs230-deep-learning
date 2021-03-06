import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

N = 100
a, b, sigma = 3, 4, 1


def get_data():
    np.random.seed(42)
    X_train = 2 * np.random.rand(N, sigma)
    y_train = b + a * X_train + np.random.randn(N, sigma)
    X_test = np.array([[0], [2]])
    y_test = b + a * X_test
    return X_train, y_train, X_test, y_test


def preprocess_data():
    X_train, y_train, X_test, y_test = get_data()
    X_train_b = np.c_[np.ones((N, 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    return X_train_b, y_train, X_test_b, y_test


def plot_regression(y_pred):
    plt.plot(X_train, y_train, 'b.')
    plt.plot(X_test, y_pred, "r-")
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$y$', rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.show()


def normal_equations():
    theta = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
    y_pred = X_test_b.dot(theta)
    print(theta)
    plot_regression(y_pred)


def fit_lr_sklearn():
    lr = LinearRegression()
    lr.fit(X_train_b, y_train)
    y_pred = lr.predict(X_test_b)
    plot_regression(y_pred)


def batch_grad_descent(n_iterations=1000, lrate=.1, m=N):
    np.random.seed(42)
    theta = np.random.randn(2, 1)
    for iteration in range(n_iterations):
        grads = (2 / m) * X_train_b.T.dot(X_train_b.dot(theta) - y_train)
        theta -= lrate * grads
    print(theta)
    y_pred = X_test_b.dot(theta)
    plot_regression(y_pred)


def stochastic_grad_descent(epochs=50, m=N):
    np.random.seed(42)
    theta = np.random.randn(2, 1)
    for epoch in range(epochs):
        for i in range(m):
            # with random index
            # random_index = np.random.randint(m)
            # x = X_train_b[random_index, :].reshape(1, 2)
            # yi = y_train[random_index]

            # with circular index
            xi = X_train_b[i, :].reshape(1, 2)
            yi = y_train[i]

            grads = 2 * (xi.dot(theta) - yi) * xi.T
            lrate = learning_schedule(epoch * m + i)
            theta -= lrate * grads
        print(f'epoch:{epoch:2d} rate:{lrate:.4f}')

    print(theta)
    y_pred = X_test_b.dot(theta)
    # plot_regression(y_pred)


def stochastic_grad_descent2(epochs=50, lrate=.1, m=N):
    np.random.seed(42)
    theta = np.random.randn(2, 1)
    for epoch in range(epochs):
        for i in range(m):
            # with circular index
            xi = X_train_b[i, :].reshape(1, 2)
            yi = y_train[i]

            grads = 2 * (xi.dot(theta) - yi) * xi.T
            theta -= lrate * grads
        lrate *= .90
        print(f'epoch:{epoch:2d} rate:{lrate:.4f}')

    print(theta)


def learning_schedule(t, t0=5, t1=50):
    return t0 / (t + t1)


def sgd_sklearn():
    sgd = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
    sgd.fit(X_train, y_train.ravel())
    print(sgd.intercept_, sgd.coef_)


def mini_batch_grad_descent(epochs=50, m=N, k=10):
    np.random.seed(42)
    theta = np.random.randn(2, 1)
    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(m)
        X_train_b_sf = X_train_b[shuffled_indices, :]
        y_train_sf = y_train[shuffled_indices, :]
        for i in range(m // k):
            xmb = X_train_b_sf[i*k:(i+1)*k, :]
            ymb = y_train_sf[i*k:(i+1)*k, :]

            # so formula here is the same as in batch GD except we
            # use mini-batch and k instead of m
            grads = (2 / k) * xmb.T.dot(xmb.dot(theta) - ymb)

            lrate = learning_schedule(epoch * m / k + i)
            theta -= lrate * grads
        print(f'epoch:{epoch:2d} rate:{lrate:.4f}')

    print(theta)
    y_pred = X_test_b.dot(theta)
    # plot_regression(y_pred)


if __name__ == '__main__':
    X_train, _, X_test, _ = get_data()
    X_train_b, y_train, X_test_b, y_test = preprocess_data()
    # fit_lr_sklearn()
    # normal_equations()
    # batch_grad_descent()
    # stochastic_grad_descent2()
    # sgd_sklearn()
    mini_batch_grad_descent()