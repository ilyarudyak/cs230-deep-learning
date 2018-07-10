import tensorflow as tf
import numpy as np
import time

from tensorflow.examples.tutorials.mnist import input_data


class SoftmaxClassifier:
    def __init__(self, data_dir='/Users/ilyarudyak/data/mnist'):
        self.mnist = input_data.read_data_sets(data_dir, one_hot=True)

        self.x = tf.placeholder(tf.float32, [None, 784])
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.y_true = tf.placeholder(tf.float32, [None, 10])
        self.y_pred = tf.matmul(self.x, self.W)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.y_pred, labels=self.y_true))

        self.gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        correct_mask = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y_true, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

    def train_loop(self, steps=1000, minibatch_size=100):
        with tf.Session() as sess:
            # train
            sess.run(tf.global_variables_initializer())
            for _ in range(steps):
                batch_xs, batch_ys = self.mnist.train.next_batch(minibatch_size)
                sess.run(self.gd_step, feed_dict={self.x: batch_xs, self.y_true: batch_ys})

            # test
            ans = sess.run(self.accuracy, feed_dict={self.x: self.mnist.validation.images,
                                                     self.y_true: self.mnist.validation.labels})

        print("accuracy:{:.4}".format(ans))


if __name__ == '__main__':
    smc = SoftmaxClassifier()
    smc.train_loop(steps=500)
