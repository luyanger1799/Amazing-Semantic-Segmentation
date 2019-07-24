"""
The implementation of learning rate scheduler.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import numpy as np


def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 100.0
    lrate = initial_lrate * np.power(drop,
           np.floor((1+epoch)/epochs_drop))
    return lrate


def poly_decay(lr=3e-4, max_epochs=100):
    def decay(epoch):
        # initialize the maximum number of epochs, base learning rate,
        # and power of the polynomial
        lrate = lr*(1-np.power(epoch/max_epochs, 0.9))
        return lrate
    return decay
