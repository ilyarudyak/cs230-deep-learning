import tensorflow as tf
import problem_unittests as tests


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    w, h, d = image_shape
    return tf.placeholder(tf.float32, shape=[None, w, h, d], name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, shape=[None, n_classes], name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name='keep_prob')


def conv2d_relu_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    conv2d = tf.nn.conv2d(x_tensor, W, conv_strides)
    conv2d_relu = tf.nn.relu(conv2d + b)
    max_pool = tf.nn.max_pool(conv2d_relu, pool_ksize, pool_strides)
    return max_pool


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    _, w, h, d = x_tensor.get_shape()
    return tf.reshape(x_tensor, [-1, w * h * d])


def fully_conn(x_tensor, num_outputs):
    in_size = int(x_tensor.get_shape()[1])
    W = weight_variable([in_size, num_outputs])
    b = bias_variable([num_outputs])
    return tf.matmul(x_tensor, W) + b


def fully_conn_relu(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    return tf.nn.relu(fully_conn(x_tensor, num_outputs))


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    return fully_conn(x_tensor, num_outputs)


if __name__ == '__main__':
    tf.reset_default_graph()
    tests.test_nn_image_inputs(neural_net_image_input)
    tests.test_nn_label_inputs(neural_net_label_input)
    tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

    tests.test_flatten(flatten)
    tests.test_fully_conn(fully_conn_relu)
    tests.test_output(output)

