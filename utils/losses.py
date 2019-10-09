"""
The implementation of somes losses based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf

backend = tf.keras.backend


def categorical_crossentropy_with_logits(y_true, y_pred):
    # compute cross entropy
    cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

    # compute loss
    loss = backend.mean(backend.sum(cross_entropy, axis=[1, 2]))
    return loss


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)
        # compute ce loss
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=False)
        # compute weights
        weights = backend.sum(alpha * backend.pow(1 - y_pred, gamma) * y_true, axis=-1)
        return backend.mean(backend.sum(weights * cross_entropy, axis=[1, 2]))

    return loss


def miou_loss(weights=None, num_classes=2):
    if weights is not None:
        weights = tf.Variable(weights, dtype=tf.float32)
    else:
        weights = tf.Variable([1] * num_classes, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)

        inter = y_pred * y_true
        inter = backend.sum(inter, axis=[1, 2])

        union = y_pred + y_true - (y_pred * y_true)
        union = backend.sum(union, axis=[1, 2])

        return -backend.mean((weights * inter) / (weights * union + 1e-8))

    return loss

