"""
The implementation of some callbacks based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf
import numpy as np

callbacks = tf.keras.callbacks
backend = tf.keras.backend


class LearningRateScheduler(callbacks.Callback):
    def __init__(self, schedule, learning_rate=None, warmup=False, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.verbose = verbose
        self.warmup = warmup
        self.warmup_steps = 200

        if warmup and learning_rate is None:
            raise ValueError('learning rate cannot be None if warmup is used.')

    def on_batch_begin(self, batch, logs=None):
        if self.warmup and batch <= self.warmup_steps:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            lr = self.learning_rate * (batch + 1) / self.warmup_steps
            backend.set_value(self.model.optimizer.lr, lr)
        else:
            self.warmup = False

    def on_epoch_begin(self, epoch, logs=None):
        if not self.warmup:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            try:  # new API
                lr = float(backend.get_value(self.model.optimizer.lr))
                lr = self.schedule(epoch, lr)
            except TypeError:  # Support for old API for backward compatibility
                lr = self.schedule(epoch)
            if not isinstance(lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function '
                                 'should be float.')
            backend.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nEpoch %05d: LearningRateScheduler reducing learning '
                      'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)
