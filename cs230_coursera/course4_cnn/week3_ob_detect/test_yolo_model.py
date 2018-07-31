import tensorflow as tf
import numpy as np

from yolo_model import yolo_filter_boxes


def test_yolo_filter_boxes():
    with tf.Session() as test_a:
        box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
        boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
        box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
        scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.5)

        assert np.isclose(10.7506, scores[2].eval())

        target = np.array([8.42653275, 3.27136683, -0.5313437, -4.94137383])
        assert np.allclose(target, boxes[2].eval())

        assert np.isclose(7, classes[2].eval())

        print(f'\n\n---> scores shape: {scores.eval().shape} {scores.eval()[:5]}')
        print(f'---> classes shape: {classes.eval().shape} {classes.eval()[:5]}')
        print(f'---> boxes shape: {boxes.eval().shape}\n\n')


