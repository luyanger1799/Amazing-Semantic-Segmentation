"""
The implementation of some metrics based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf


def mean_iou(y_true, y_pred):
    num_classes = y_pred.get_shape().as_list()[-1]

    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    ious = list()
    for i in range(num_classes):
        yti = tf.equal(y_true, i)
        ypi = tf.equal(y_pred, i)

        intersection = tf.reduce_sum(tf.cast(tf.logical_and(yti, ypi), tf.float32), axis=[1, 2])
        union = tf.reduce_sum(tf.cast(tf.logical_or(yti, ypi), tf.float32), axis=[1, 2])

        iou = tf.where(tf.equal(union, 0.), tf.ones_like(union), intersection / union)
        ious.append(iou)
    ious = tf.stack(ious, axis=-1)
    return tf.reduce_sum(ious, axis=-1) / num_classes
