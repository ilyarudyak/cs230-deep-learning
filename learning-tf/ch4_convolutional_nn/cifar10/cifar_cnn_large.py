import time

from layers import *
from data_manager import *


class ConvNetCIFAR10Large:
    def __init__(self,
                 adam_learning_rate=1e-3):
        self.cifar10 = CifarDataManager()
        self.adam_learning_rate = adam_learning_rate

        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])  # size of 3D images in CIFAR10
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])  # still 10 classes

        # we need a placeholder here in order to change keep_prob
        # from .5 for training to 1.0 for testing
        self.keep_prob = tf.placeholder(tf.float32)

        self.build_conv_net()

    # noinspection PyAttributeOutsideInit
    def build_conv_net(self):

        ############ start of NEW code ############

        C1, C2, C3 = 32, 64, 128
        F1 = 600

        conv1_1 = conv_layer(self.x, shape=[3, 3, 3, C1])
        conv1_2 = conv_layer(conv1_1, shape=[3, 3, C1, C1])
        conv1_3 = conv_layer(conv1_2, shape=[3, 3, C1, C1])
        conv1_pool = max_pool_2x2(conv1_3)
        conv1_drop = tf.nn.dropout(conv1_pool, keep_prob=self.keep_prob)

        conv2_1 = conv_layer(conv1_drop, shape=[3, 3, C1, C2])
        conv2_2 = conv_layer(conv2_1, shape=[3, 3, C2, C2])
        conv2_3 = conv_layer(conv2_2, shape=[3, 3, C2, C2])
        conv2_pool = max_pool_2x2(conv2_3)
        conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=self.keep_prob)

        conv3_1 = conv_layer(conv2_drop, shape=[3, 3, C2, C3])
        conv3_2 = conv_layer(conv3_1, shape=[3, 3, C3, C3])
        conv3_3 = conv_layer(conv3_2, shape=[3, 3, C3, C3])
        conv3_pool = tf.nn.max_pool(conv3_3, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
        conv3_flat = tf.reshape(conv3_pool, [-1, C3])
        conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=self.keep_prob)

        full1 = tf.nn.relu(full_layer(conv3_drop, F1))
        full1_drop = tf.nn.dropout(full1, keep_prob=self.keep_prob)

        y_conv = full_layer(full1_drop, 10)

        ############ end of NEW code ############

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_conv, labels=self.y_))
        self.train_step = tf.train.AdamOptimizer(self.adam_learning_rate).minimize(self.cross_entropy)
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
    cnn = ConvNetCIFAR10Large()
    cnn.train_loop(steps=500)
