import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ch9_intro_tf.reset import reset_graph


def get_linear_regression_data():
    x_data = np.random.randn(2000, 3)
    w_real = [0.3, 0.5, 0.1]
    b_real = -0.2

    noise = np.random.randn(1, 2000) * 0.1
    y_data = np.matmul(w_real, x_data.T) + b_real + noise

    # plt.plot(np.matmul(w_real, x_data.T) + b_real, y_data.flatten(), 'o')
    # plt.show()

    return x_data, y_data


def fit_linear_regression(n_epoch=10, learning_rate=0.5):
    x_data, y_data = get_linear_regression_data()

    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)
    w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weights')
    b = tf.Variable(0, dtype=tf.float32, name='bias')
    y_pred = tf.matmul(w, tf.transpose(x)) + b
    loss = tf.reduce_mean(tf.square(y_true - y_pred))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    wb_ = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(n_epoch+1):
            sess.run(train, {x: x_data, y_true: y_data})
            if step % 5 == 0:
                print(step, sess.run([w, b]))
                wb_.append(sess.run([w, b]))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_logistic_regression_data(n=20000):
    x_data = np.random.randn(n, 3)
    w_real = [0.3, 0.5, 0.1]
    b_real = -0.2
    wxb = np.matmul(w_real, x_data.T) + b_real

    y_data_pre_noise = sigmoid(wxb)
    y_data = np.random.binomial(1, y_data_pre_noise)

    return x_data, y_data


def fit_logistic_regression(learning_rate=0.5):
    x_data, y_data = get_logistic_regression_data()

    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)

    w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weights')
    b = tf.Variable(0, dtype=tf.float32, name='bias')
    y_pred = tf.matmul(w, tf.transpose(x)) + b

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = tf.reduce_mean(loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    NUM_STEPS = 50
    wb_ = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train, {x: x_data, y_true: y_data})
            if step % 5 == 0:
                print(step, sess.run([w, b]))
                wb_.append(sess.run([w, b]))

        print(50, sess.run([w, b]))


if __name__ == '__main__':
    reset_graph()
    sns.set()

    # fit_linear_regression()
    fit_logistic_regression()



