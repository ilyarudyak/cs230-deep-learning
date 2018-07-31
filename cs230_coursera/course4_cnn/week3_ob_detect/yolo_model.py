import tensorflow as tf


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold],
    then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as
    it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes thanks to the max box_scores,
    # keep track of the corresponding score
    box_classes = tf.argmax(box_scores, axis=-1)
    box_class_scores = tf.reduce_max(box_scores, axis=-1)

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold".
    # The mask should have the same dimension as box_class_scores, and be True
    # for the boxes you want to keep (with probability >= threshold)
    mask = box_class_scores >= threshold

    # Step 4: Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores, mask)
    boxes = tf.boolean_mask(boxes, mask)
    classes = tf.boolean_mask(box_classes, mask)

    return scores, boxes, classes


def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """
    intersection_area = area(intersection(box1, box2))
    union_area = area(box1) + area(box2) - intersection_area
    return intersection_area / union_area


def area(box):
    x1, y1, x2, y2 = box
    return (y2 - y1) * (x2 - x1)


def intersection(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    return max(x1, x1_), max(y1, y1_), min(x2, x2_), min(y2, y2_)


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    # tensor to be used in tf.image.non_max_suppression()
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    # initialize variable max_boxes_tensor
    tf.get_default_session().run(tf.variables_initializer([max_boxes_tensor]))

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores,
                                               max_output_size=max_boxes,
                                               iou_threshold=iou_threshold)

    # Use tf.gather() to select only nms_indices from scores, boxes and classes
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    return scores, boxes, classes


if __name__ == '__main__':
    with tf.Session() as test_b:
        scores = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
        boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed=1)
        classes = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
        scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
        print(scores.eval())
