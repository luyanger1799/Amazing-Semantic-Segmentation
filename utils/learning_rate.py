"""
The implementation of learning rate scheduler.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import numpy as np


def step_decay(lr=3e-4, max_epochs=100):
    """
    step decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :return: current lr
    """
    drop = 0.1

    def decay(epoch):
        lrate = lr * np.power(drop, np.floor((1 + epoch) / max_epochs))
        return lrate

    return decay


def poly_decay(lr=3e-4, max_epochs=100):
    """
    poly decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :return: current lr
    """

    def decay(epoch):
        lrate = lr * (1 - np.power(epoch / max_epochs, 0.9))
        return lrate

    return decay


def cosine_decay(max_epochs, max_lr, min_lr=1e-7):
    """
    cosine annealing scheduler.
    :param max_epochs: max epochs
    :param max_lr: max lr
    :param min_lr: min lr
    :return: current lr
    """

    def decay(epoch):
        lrate = min_lr + (max_lr - min_lr) * (
                1 + np.cos(np.pi * epoch / max_epochs)) / 2
        return lrate

    return decay

