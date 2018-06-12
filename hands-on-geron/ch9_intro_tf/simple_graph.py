import tensorflow as tf
import numpy as np

from ch9_intro_tf.reset import reset_graph


def build_simple_graph():
    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f = x * x * y + y + 2

    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()

    return result


if __name__ == '__main__':
    reset_graph()
    print(f'result={build_simple_graph()}')
