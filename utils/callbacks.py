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
    def __init__(self,
                 schedule,
                 learning_rate=None,
                 warmup=False,
                 steps_per_epoch=None,
                 verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.verbose = verbose
        self.warmup_epochs = 5 if warmup else 0
        self.warmup_steps = int(steps_per_epoch) * self.warmup_epochs if warmup else 0
        self.global_batch = 0

        if warmup and learning_rate is None:
            raise ValueError('learning_rate cannot be None if warmup is used.')
        if warmup and steps_per_epoch is None:
            raise ValueError('steps_per_epoch cannot be None if warmup is used.')

    def on_train_batch_begin(self, batch, logs=None):
        self.global_batch += 1
        if self.global_batch < self.warmup_steps:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            lr = self.learning_rate * self.global_batch / self.warmup_steps
            backend.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: LearningRateScheduler warming up learning '
                      'rate to %s.' % (self.global_batch, lr))

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(backend.get_value(self.model.optimizer.lr))

        if epoch >= self.warmup_epochs:
            try:  # new API
                lr = self.schedule(epoch - self.warmup_epochs, lr)
            except TypeError:  # Support for old API for backward compatibility
                lr = self.schedule(epoch - self.warmup_epochs)
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
