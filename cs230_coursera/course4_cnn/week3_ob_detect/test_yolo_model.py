import tensorflow as tf
import numpy as np

from yolo_model import yolo_filter_boxes, iou


def test_yolo_filter_boxes():
    with tf.Session() as test_a:
        box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
        boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
        box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
        scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.5)

        scores_val = scores.eval()
        boxes_val = boxes.eval()
        classes_val = classes.eval()

        assert np.isclose(10.7506, scores_val[2])

        target = np.array([8.42653275, 3.27136683, -0.5313437, -4.94137383])
        assert np.allclose(target, boxes_val[2])

        assert np.isclose(7, classes_val[2])

        print(f'\n\n---> scores shape: {scores_val.shape} {scores_val[:5]}')
        print(f'---> classes shape: {classes_val.shape} {classes_val[:5]}')
        print(f'---> boxes shape: {boxes_val.shape}\n\n')


def test_iou():
    box1 = (2, 1, 4, 3)
    box2 = (1, 2, 3, 4)

    assert np.isclose(0.142857142857, iou(box1, box2))
