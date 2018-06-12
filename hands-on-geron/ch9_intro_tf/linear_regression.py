import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing

from ch9_intro_tf.reset import reset_graph


def get_housing_data():
    housing = fetch_california_housing()
    m, n = housing.data.shape
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
    return housing_data_plus_bias, housing.target.reshape(-1, 1)


def fit_regression_with_normal_eq():
    X_train, y_train = get_housing_data()

    X = tf.constant(X_train, dtype=tf.float32, name="X")
    y = tf.constant(y_train, dtype=tf.float32, name="y")
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

    with tf.Session() as sess:
        theta_value = theta.eval()

    return theta_value


if __name__ == '__main__':
    reset_graph()
    print(f'theta={fit_regression_with_normal_eq()}')
