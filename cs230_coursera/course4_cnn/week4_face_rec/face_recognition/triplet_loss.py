import tensorflow as tf


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    # Step 1: Compute the (encoding) distance between the anchor and the positive,
    # you will need to sum over axis=-1
    pos_dist = tf.reduce_sum((anchor - positive) ** 2, axis=-1)

    # Step 2: Compute the (encoding) distance between the anchor and the negative,
    # you will need to sum over axis=-1
    neg_dist = tf.reduce_sum((anchor - negative) ** 2, axis=-1)

    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha

    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss
