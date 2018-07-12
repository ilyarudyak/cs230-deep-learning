import tensorflow as tf
import problem_unittests as tests


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


def conv2d_maxpool2(x_tensor, parameters):
    conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides = parameters

    weight = tf.Variable(tf.random_normal([conv_ksize[0], conv_ksize[1], \
                                           x_tensor.get_shape().as_list()[3], conv_num_outputs], stddev=0.1))

    bias = tf.Variable(tf.random_normal([conv_num_outputs], stddev=0.1))

    x_tensor = tf.nn.conv2d(x_tensor, weight, [1, conv_strides[0], conv_strides[1], 1], \
                            padding='SAME')

    x_tensor = tf.nn.bias_add(x_tensor, bias)

    x_tensor = tf.nn.relu(x_tensor)

    x_tensor = tf.nn.max_pool(x_tensor, [1, pool_ksize[0], pool_ksize[1], 1], \
                              [1, pool_strides[0], pool_strides[1], 1], padding='SAME')

    return x_tensor


def conv2d_maxpool(x_tensor, parameters):
    conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides = parameters
    conv2d_layer = tf.contrib.layers.conv2d(x_tensor, conv_num_outputs,
                                            kernel_size=conv_ksize,
                                            stride=conv_strides)
    maxpool_layer = tf.contrib.layers.max_pool2d(conv2d_layer,
                                                 kernel_size=pool_ksize,
                                                 stride=pool_strides,
                                                 padding='SAME')

    # maxpool_layer = tf.nn.max_pool(conv2d_layer, [1, pool_ksize[0], pool_ksize[1], 1], \
    #                           [1, pool_strides[0], pool_strides[1], 1], padding='SAME')
    return maxpool_layer


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    return tf.contrib.layers.flatten(x_tensor)


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # default params:
    # activation_fn=tf.nn.relu,
    # weights_initializer=initializers.xavier_initializer(),
    #
    return tf.contrib.layers.fully_connected(x_tensor, num_outputs)


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    return tf.contrib.layers.fully_connected(x_tensor, num_outputs, activation_fn=None)


def dropout(x_tensor, keep_prob):
    return tf.nn.dropout(x_tensor, keep_prob)


