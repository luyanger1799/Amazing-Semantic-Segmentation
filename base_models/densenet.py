"""
The implementation of DenseNet121/169/201/264 based on Tensorflow.
Some codes are based on official tensorflow source codes.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from utils.layers import Concatenate
import tensorflow as tf

layers = tf.keras.layers
backend = tf.keras.backend


class DenseNet(object):
    def __init__(self, version='DenseNet121', dilation=None, **kwargs):
        """
        The implementation of DenseNet based on Tensorflow.
        :param version: 'DenseNet121', 'DenseNet169', 'DenseNet201' or 'DenseNet264'.
        :param dilation: Whether to use dilation strategy.
        :param kwargs: other parameters.
        """
        super(DenseNet, self).__init__(**kwargs)
        params = {'DenseNet121': [6, 12, 24, 16],
                  'DenseNet169': [6, 12, 32, 32],
                  'DenseNet201': [6, 12, 48, 32],
                  'DenseNet264': [6, 12, 64, 48]}
        self.version = version
        assert version in params
        self.params = params[version]

        if dilation is None:
            self.dilation = [1, 1]
        else:
            self.dilation = dilation
        assert len(self.dilation) == 2

    def _dense_block(self, x, blocks, name, dilation=1):
        """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = self._conv_block(x, 32, name=name + '_block' + str(i + 1), dilation=dilation)
        return x

    def _transition_block(self, x, reduction, name, dilation=1):
        """A transition block.

        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_bn')(x)
        x = layers.Activation('relu', name=name + '_relu')(x)
        x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                          use_bias=False,
                          name=name + '_conv',
                          dilation_rate=dilation)(x)
        if dilation == 1:
            x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    def _conv_block(self, x, growth_rate, name, dilation=1):
        """A building block for a dense block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            Output tensor for the block.
        """
        _, h, w, _ = backend.int_shape(x)

        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x1 = layers.BatchNormalization(axis=bn_axis,
                                       epsilon=1.001e-5,
                                       name=name + '_0_bn')(x)
        x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
        x1 = layers.Conv2D(4 * growth_rate, 1,
                           use_bias=False,
                           name=name + '_1_conv')(x1)
        x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_1_bn')(x1)
        x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
        x1 = layers.Conv2D(growth_rate, 3,
                           padding='same',
                           use_bias=False,
                           name=name + '_2_conv',
                           dilation_rate=dilation)(x1)
        x = Concatenate(out_size=(h, w), axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    def __call__(self, inputs, output_stages='c5', **kwargs):
        """
        call for DenseNet.
        :param inputs: a 4-D tensor.
        :param output_stages: str or a list of str containing the output stages.
        :param kwargs: other parameters.
        :return: the output of different stages.
        """
        _, h, w, _ = backend.int_shape(inputs)

        blocks = self.params
        dilation = self.dilation
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
        x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
        x = layers.Activation('relu', name='conv1/relu')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)
        c1 = x

        x = self._dense_block(x, blocks[0], name='conv2')
        x = self._transition_block(x, 0.5, name='pool2')
        c2 = x

        x = self._dense_block(x, blocks[1], name='conv3')
        x = self._transition_block(x, 0.5, name='pool3', dilation=dilation[0])
        c3 = x

        x = self._dense_block(x, blocks[2], name='conv4', dilation=dilation[0])
        x = self._transition_block(x, 0.5, name='pool4', dilation=dilation[1])
        c4 = x

        x = self._dense_block(x, blocks[3], name='conv5', dilation=dilation[1])
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = layers.Activation('relu', name='relu')(x)
        c5 = x

        self.outputs = {'c1': c1,
                        'c2': c2,
                        'c3': c3,
                        'c4': c4,
                        'c5': c5}

        if type(output_stages) is not list:
            return self.outputs[output_stages]
        else:
            return [self.outputs[ci] for ci in output_stages]
