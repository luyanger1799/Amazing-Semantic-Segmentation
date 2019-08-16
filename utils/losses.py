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
