"""
The implementation of VGG16/VGG19 based on Tensorflow.
Some codes are based on official tensorflow source codes.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf

layers = tf.keras.layers
backend = tf.keras.backend


class VGG(object):
    def __init__(self, version='VGG16', dilation=None, **kwargs):
        """
        The implementation of VGG16 and VGG19 based on Tensorflow.
        :param version: 'VGG16' or 'VGG19'
        :param dilation: Whether to use dilation strategy
        :param kwargs: other parameters.
        """
        super(VGG, self).__init__(**kwargs)
        params = {'VGG16': [2, 2, 3, 3, 3],
                  'VGG19': [2, 2, 4, 4, 4]}
        self.version = version
        assert version in params
        self.params = params[version]

        if dilation is None:
            self.dilation = [1, 1]
        else:
            self.dilation = dilation
        assert len(self.dilation) == 2

    def __call__(self, inputs, output_stages='c5', **kwargs):
        """
        call for VGG16 or VGG19.
        :param inputs: a 4-D tensor.
        :param output_stages: str or a list of str containing the output stages.
        :param kwargs: other parameters.
        :return: the output of different stages.
        """
        dilation = self.dilation
        _, h, w, _ = backend.int_shape(inputs)

        # Block 1
        for i in range(self.params[0]):
            x = layers.Conv2D(64, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block1_conv' + str(i + 1))(inputs)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        c1 = x

        # Block 2
        for i in range(self.params[1]):
            x = layers.Conv2D(128, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block2_conv' + str(i + 1))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        c2 = x

        # Block 3
        for i in range(self.params[2]):
            x = layers.Conv2D(256, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block3_conv' + str(i + 1))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        c3 = x

        # Block 4
        for i in range(self.params[3]):
            x = layers.Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block4_conv' + str(i + 1),
                              dilation_rate=dilation[0])(x)
        if dilation[0] == 1:
            x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        c4 = x

        # Block 5
        for i in range(self.params[4]):
            x = layers.Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block5_conv' + str(i + 1),
                              dilation_rate=dilation[1])(x)
        if dilation[1] == 1:
            x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
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
