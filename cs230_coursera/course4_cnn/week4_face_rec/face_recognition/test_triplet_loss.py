import tensorflow as tf
import numpy as np

from triplet_loss import triplet_loss


def test_triplet_loss():
    with tf.Session() as test:
        tf.set_random_seed(1)
        y_true = (None, None, None)
        y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
                  tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
                  tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
        loss = triplet_loss(y_true, y_pred)
        loss_val = loss.eval()

        assert np.isclose(528.143, loss_val)


if __name__ == '__main__':
    test_triplet_loss()
