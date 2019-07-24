"""
The implementation of somes losses based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf

backend = tf.keras.backend


def categorical_crossentropy_logits(y_true, y_pred):

    return backend.categorical_crossentropy(y_true, y_pred, from_logits=True)
