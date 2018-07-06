import tensorflow as tf
import numpy as np
import time

from tensorflow.examples.tutorials.mnist import input_data


class ConvNetMNIST:
    def __init__(self,
                 adam_learning_rate=1e-4,
                 data_dir='/Users/ilyarudyak/data/mnist'):
        self.mnist = input_data.read_data_sets(data_dir, one_hot=True)
        self.adam_learning_rate = adam_learning_rate

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.keep_prob = tf.placeholder(tf.float32)

        self.build_conv_net()

    # noinspection PyAttributeOutsideInit
    def build_conv_net(self):

        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
        conv1_pool = max_pool_2x2(conv1)

        conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
        conv2_pool = max_pool_2x2(conv2)

        conv2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])
        full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

        full1_drop = tf.nn.dropout(full_1, keep_prob=self.keep_prob)

        y_conv = full_layer(full1_drop, 10)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_conv, labels=self.y_))
        self.train_step = tf.train.AdamOptimizer(self.adam_learning_rate).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train_loop(self, steps=2500, minibatch_size=50):
        start = time.time()
        epoch_iteration = self.mnist.train.num_examples / minibatch_size
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(steps):
                train_batch = self.mnist.train.next_batch(minibatch_size)
                val_batch = self.mnist.validation.next_batch(minibatch_size)

                if i % epoch_iteration == 0:
                    print("======> epoch:{} starting ...".format(int(i / epoch_iteration)))

                if i % 100 == 0:
                    train_accuracy = sess.run(self.accuracy, feed_dict={self.x: train_batch[0],
                                                                        self.y_: train_batch[1],
                                                                        self.keep_prob: 1.0})
                    validation_accuracy = sess.run(self.accuracy, feed_dict={self.x: val_batch[0],
                                                                             self.y_: val_batch[1],
                                                                             self.keep_prob: 1.0})
                    print("======> step {:4d}: training accuracy:{:.4f} validation accuracy:{:.4f}"
                          .format(i, train_accuracy, validation_accuracy))

                sess.run(self.train_step, feed_dict={self.x: train_batch[0],
                                                     self.y_: train_batch[1],
                                                     self.keep_prob: 0.5})

            X = self.mnist.test.images.reshape(10, 1000, 784)
            Y = self.mnist.test.labels.reshape(10, 1000, 10)
            test_accuracy = np.mean(
                [sess.run(self.accuracy, feed_dict={self.x: X[i],
                                                    self.y_: Y[i],
                                                    self.keep_prob: 1.0})
                 for i in range(10)])

        print("*** test accuracy: {:.4f} elapsed time: {:.1f}s ***".format(test_accuracy, time.time() - start))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b


if __name__ == '__main__':
    cnn = ConvNetMNIST()
    cnn.train_loop(steps=500)
