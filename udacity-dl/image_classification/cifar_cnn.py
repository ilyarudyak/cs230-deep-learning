import time
import numpy as np

from data_manager import CifarDataManager
from layers import *


# noinspection PyAttributeOutsideInit
class ConvNetCIFAR10:
    def __init__(self, image_shape=(32, 32, 3), n_classes=10):
        self.cifar10 = CifarDataManager()
        self.x = neural_net_image_input(image_shape)
        self.y_ = neural_net_label_input(n_classes)
        self.keep_prob = neural_net_keep_prob_input()
        self.n_classes = n_classes

        self.set_params()
        self.build_conv_net()

    def set_params(self):
        self.conv_ksize = (8, 8)
        self.conv_strides = (self.conv_ksize[0] / 2, self.conv_ksize[1] / 2)  # (4, 4)
        self.conv_num_outputs = self.conv_ksize[0] * self.conv_ksize[1]  # 64

        self.pool_ksize = (self.conv_strides[0], self.conv_strides[1])  # (4, 4)
        self.pool_strides = (self.pool_ksize[0] / 2, self.pool_ksize[1] / 2)  # (2, 2)

        self.num_outputs = self.n_classes

        self.parameters = [self.conv_num_outputs, self.conv_ksize, self.conv_strides,
                           self.pool_ksize, self.pool_strides]

    def build_conv_net(self):
        conv2d_maxpool_layer_1 = conv2d_maxpool(self.x, self.parameters)
        conv2d_maxpool_layer_2 = conv2d_maxpool(conv2d_maxpool_layer_1, self.parameters)

        flatten_layer = flatten(conv2d_maxpool_layer_2)
        flatten_dropout_layer = dropout(flatten_layer, self.keep_prob)

        fully_conn_layer = fully_conn(flatten_dropout_layer, self.num_outputs * 2)
        fully_conn_dropout_layer = dropout(fully_conn_layer, self.keep_prob)

        y_conv = output(fully_conn_dropout_layer, self.num_outputs)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_conv, labels=self.y_))
        self.train_step = tf.train.AdamOptimizer().minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train_loop(self, steps=2500, minibatch_size=50):
        start = time.time()
        epoch_iteration = self.cifar10.train.num_examples / minibatch_size
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(steps):
                train_batch = self.cifar10.train.next_batch(minibatch_size)
                val_batch = self.cifar10.test.next_batch(minibatch_size)

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

            X = self.cifar10.test.images.reshape(10, 1000, 32, 32, 3)
            Y = self.cifar10.test.labels.reshape(10, 1000, 10)
            test_accuracy = np.mean(
                [sess.run(self.accuracy, feed_dict={self.x: X[i],
                                                    self.y_: Y[i],
                                                    self.keep_prob: 1.0})
                 for i in range(10)])

        print("*** test accuracy: {:.4f} elapsed time: {:.1f}s ***".format(test_accuracy, time.time() - start))


if __name__ == '__main__':
    tf.reset_default_graph()
    cnn = ConvNetCIFAR10()
    cnn.train_loop(steps=500)
