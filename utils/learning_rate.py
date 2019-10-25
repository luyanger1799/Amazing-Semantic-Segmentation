"""
The implementation of learning rate scheduler.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import numpy as np


def step_decay(lr=3e-4, max_epochs=100, warmup_steps=0):
    """
    step decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :param warmup_steps: warm up steps
    :return: current lr
    """
    def decay(epoch):
        drop = 0.1
        if epoch <= warmup_steps:
            lrate = lr * epoch / warmup_steps
        else:
            lrate = lr * np.power(drop, np.floor((1 + (epoch - warmup_steps)) / (max_epochs - warmup_steps)))
        return lrate

    return decay


def poly_decay(lr=3e-4, max_epochs=100, warmup_steps=0):
    """
    poly decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :param warmup_steps: warm up steps
    :return: current lr
    """
    def decay(epoch):
        if epoch <= warmup_steps:
            lrate = lr * epoch / warmup_steps
        else:
            lrate = lr * (1 - np.power((epoch - warmup_steps) / (max_epochs - warmup_steps), 0.9))
        return lrate

    return decay


def cosine_decay(max_epochs, max_lr, min_lr=1e-7, warmup_steps=0):
    """
    cosine annealing scheduler.
    :param max_epochs: max epochs
    :param max_lr: max lr
    :param min_lr: min lr
    :param warmup_steps: warm up steps
    :return: current lr
    """

    def decay(epoch):
        if epoch <= warmup_steps:
            lrate = max_lr * epoch / warmup_steps
        else:
            lrate = min_lr + (max_lr - min_lr) * (
                    1 + np.cos(np.pi * (epoch - warmup_steps) / (max_epochs - warmup_steps))) / 2
        return lrate

    return decay
