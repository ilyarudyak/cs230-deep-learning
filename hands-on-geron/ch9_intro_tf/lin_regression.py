import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

from ch9_intro_tf.reset import reset_graph


def get_housing_data():
    housing = fetch_california_housing()
    m, n = housing.data.shape
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
    return housing_data_plus_bias, housing.target.reshape(-1, 1)


def get_scaled_data():
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
    return scaled_housing_data_plus_bias, housing.target.reshape(-1, 1)


def fit_regression_with_normal_eq():
    X_train, y_train = get_housing_data()

    X = tf.constant(X_train, dtype=tf.float32, name="X")
    y = tf.constant(y_train, dtype=tf.float32, name="y")
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

    with tf.Session() as sess:
        theta_value = theta.eval()

    return theta_value


def manual_batch_gd(learning_rate=0.01):
    mse, theta = set_variables()

    # using formula to calculate gradients
    # gradients = 2/m * tf.matmul(tf.transpose(X), error)

    # using tensorflow autodiff - we need to change just 1 line
    gradients = tf.gradients(mse, [theta])[0]

    # noinspection PyTypeChecker
    training_op = tf.assign(theta, theta - learning_rate * gradients)

    best_theta = train_model(mse, training_op, theta)

    return best_theta


def auto_batch_gd(learning_rate=0.01):
    mse, theta = set_variables()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    best_theta = train_model(mse, training_op, theta)

    return best_theta


def train_model(mse, training_op, theta, n_epochs=1000):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print(f"epoch={epoch:3d} MSE={mse.eval():.4f}")
            sess.run(training_op)

        best_theta = theta.eval()

    return best_theta


def set_variables():
    X_train, y_train = get_scaled_data()
    m, n = X_train.shape

    X = tf.constant(X_train, dtype=tf.float32, name="X")
    y = tf.constant(y_train, dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    return mse, theta


if __name__ == '__main__':
    reset_graph()
    print(f'\ntheta=\n{auto_batch_gd()}')
