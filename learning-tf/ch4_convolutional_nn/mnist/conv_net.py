import tensorflow as tf
import numpy as np
import time

from tensorflow.examples.tutorials.mnist import input_data
from ch4_convolutional_nn.layers import conv_layer, max_pool_2x2, full_layer


class ConvNetMNIST:
    def __init__(self,
                 adam_learning_rate=1e-4,
                 data_dir='/Users/ilyarudyak/data/mnist'):
        self.mnist = input_data.read_data_sets(data_dir, one_hot=True)
        self.adam_learning_rate = adam_learning_rate
        self.build_conv_net()

    # noinspection PyAttributeOutsideInit
    def build_conv_net(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
        conv1_pool = max_pool_2x2(conv1)

        conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
        conv2_pool = max_pool_2x2(conv2)

        conv2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])
        full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

        self.keep_prob = tf.placeholder(tf.float32)
        full1_drop = tf.nn.dropout(full_1, keep_prob=self.keep_prob)

        y_conv = full_layer(full1_drop, 10)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_conv, labels=self.y_))
        self.train_step = tf.train.AdamOptimizer(self.adam_learning_rate).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train_loop(self, steps=5000, minibatch_size=50):
        start = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(steps):
                batch = self.mnist.train.next_batch(minibatch_size)

                if i % 100 == 0:
                    train_accuracy = sess.run(self.accuracy, feed_dict={self.x: batch[0],
                                                                        self.y_: batch[1],
                                                                        self.keep_prob: 1.0})
                    print("======> step {:4d}: training accuracy {:.4f}".format(i, train_accuracy))

                sess.run(self.train_step, feed_dict={self.x: batch[0],
                                                     self.y_: batch[1],
                                                     self.keep_prob: 0.5})

            X = self.mnist.test.images.reshape(10, 1000, 784)
            Y = self.mnist.test.labels.reshape(10, 1000, 10)
            test_accuracy = np.mean(
                [sess.run(self.accuracy, feed_dict={self.x: X[i],
                                                    self.y_: Y[i],
                                                    self.keep_prob: 1.0})
                 for i in range(10)])

        print("*** test accuracy: {:.4f} elapsed time: {:.1f}s ***".format(test_accuracy, time.time() - start))


if __name__ == '__main__':
    cnn = ConvNetMNIST()
    cnn.train_loop(steps=500)
