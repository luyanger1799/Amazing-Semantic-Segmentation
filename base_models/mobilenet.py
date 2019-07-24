"""
The implementation of MobileNetV1/V2 based on Tensorflow.
Some codes are based on official tensorflow source codes.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf

layers = tf.keras.layers
backend = tf.keras.backend


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


class MobileNet(object):
    def __init__(self, version='MobileNetV2', dilation=None, **kwargs):
        """
        The implementation of MobileNetV1 and MobileNetV2 based on Tensorflow.
        :param version: 'MobileNetV1' or 'MobileNetV2'
        :param dilation: Whether to use dialtion strategy
        :param kwargs: other parameters
        """
        super(MobileNet, self).__init__(**kwargs)
        self.version = version
        self.mobilenet = {'MobileNetV1': self._mobilenet_v1,
                          'MobileNetV2': self._mobilenet_v2}
        assert version in self.mobilenet

        if dilation is None:
            self.dilation = [1, 1]
        else:
            self.dilation = dilation
        assert len(self.dilation) == 2

    def __call__(self, inputs, output_stages='c5', **kwargs):
        """
        call for MobileNetV1 or MobileNetV2.
        :param inputs: a 4-D tensor
        :param output_stages: str or a list of str indicating the output stages.
        :param kwargs: other parameters
        :return: a 4-D tensor
        """
        net = self.mobilenet[self.version]
        c1, c2, c3, c4, c5 = net(inputs)

        self.outputs = {'c1': c1,
                        'c2': c2,
                        'c3': c3,
                        'c4': c4,
                        'c5': c5}

        if type(output_stages) is not list:
            return self.outputs[output_stages]
        else:
            return [self.outputs[ci] for ci in output_stages]

    def _inverted_res_block_v2(self, inputs, expansion, stride, alpha, filters, block_id, dilation=1):
        """
        inverted residual block in MobileNetV2.
        :param inputs: a 4-D tensor
        :param expansion: the expansion rate.
        :param stride: stride for convolution
        :param alpha: controls the width of the network.
        :param filters: output filters
        :param block_id: block id
        :param dilation: dilation rate
        :return: a 4-D tensor
        """
        channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

        in_channels = backend.int_shape(inputs)[channel_axis]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        x = inputs
        prefix = 'block_{}_'.format(block_id)

        if block_id:
            # Expand
            x = layers.Conv2D(expansion * in_channels,
                              kernel_size=1,
                              padding='same',
                              use_bias=False,
                              activation=None,
                              name=prefix + 'expand')(x)
            x = layers.BatchNormalization(axis=channel_axis,
                                          epsilon=1e-3,
                                          momentum=0.999,
                                          name=prefix + 'expand_BN')(x)
            x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
        else:
            prefix = 'expanded_conv_'

        # Depthwise
        if stride == 2 and dilation == 1:
            x = layers.ZeroPadding2D(padding=_correct_pad(backend, x, 3),
                                     name=prefix + 'pad')(x)
        x = layers.DepthwiseConv2D(kernel_size=3,
                                   strides=stride if dilation == 1 else 1,
                                   activation=None,
                                   use_bias=False,
                                   padding='valid' if stride == 2 and dilation == 1 else 'same',
                                   name=prefix + 'depthwise',
                                   dilation_rate=dilation)(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'depthwise_BN')(x)

        x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

        # Project
        x = layers.Conv2D(pointwise_filters,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'project')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'project_BN')(x)

        if in_channels == pointwise_filters and stride == 1:
            return layers.Add(name=prefix + 'add')([inputs, x])
        return x

    def _conv_block_v1(self, inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
        """Adds an initial convolution layer (with batch normalization and relu6).

        # Arguments
            inputs: Input tensor of shape `(rows, cols, 3)`
                (with `channels_last` data format) or
                (3, rows, cols) (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(224, 224, 3)` would be one valid value.
            filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution
                along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.

        # Input shape
            4D tensor with shape:
            `(samples, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(samples, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(samples, filters, new_rows, new_cols)`
            if data_format='channels_first'
            or 4D tensor with shape:
            `(samples, new_rows, new_cols, filters)`
            if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
            Output tensor of block.
        """
        channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
        filters = int(filters * alpha)
        x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
        x = layers.Conv2D(filters, kernel,
                          padding='valid',
                          use_bias=False,
                          strides=strides,
                          name='conv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
        return layers.ReLU(6., name='conv1_relu')(x)

    def _depthwise_conv_block_v1(self, inputs, pointwise_conv_filters, alpha,
                                 depth_multiplier=1, strides=(1, 1), block_id=1, dilation=1):
        """Adds a depthwise convolution block.

        A depthwise convolution block consists of a depthwise conv,
        batch normalization, relu6, pointwise convolution,
        batch normalization and relu6 activation.

        # Arguments
            inputs: Input tensor of shape `(rows, cols, channels)`
                (with `channels_last` data format) or
                (channels, rows, cols) (with `channels_first` data format).
            pointwise_conv_filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the pointwise convolution).
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            depth_multiplier: The number of depthwise convolution output channels
                for each input channel.
                The total number of depthwise convolution output
                channels will be equal to `filters_in * depth_multiplier`.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution
                along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            block_id: Integer, a unique identification designating
                the block number.

        # Input shape
            4D tensor with shape:
            `(batch, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(batch, filters, new_rows, new_cols)`
            if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, new_rows, new_cols, filters)`
            if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
            Output tensor of block.
        """
        channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        strides = (1, 1) if dilation > 1 else strides

        if strides == (1, 1):
            x = inputs
        else:
            x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                     name='conv_pad_%d' % block_id)(inputs)
        x = layers.DepthwiseConv2D((3, 3),
                                   padding='same' if strides == (1, 1) else 'valid',
                                   depth_multiplier=depth_multiplier,
                                   strides=strides,
                                   use_bias=False,
                                   name='conv_dw_%d' % block_id,
                                   dilation_rate=dilation)(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
        x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

        x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                          padding='same',
                          use_bias=False,
                          strides=(1, 1),
                          name='conv_pw_%d' % block_id)(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name='conv_pw_%d_bn' % block_id)(x)
        return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

    def _mobilenet_v1(self, inputs, alpha=1.0, depth_multiplier=1):
        """
        call for MobileNetV1.
        :param inputs: a 4-D tensor.
        :param alpha: controls the width of the network.
        :param depth_multiplier: depth multiplier for depthwise convolution.
        :return: .
        """
        dilation = self.dilation

        x = self._conv_block_v1(inputs, 32, alpha, strides=(2, 2))
        x = self._depthwise_conv_block_v1(x, 64, alpha, depth_multiplier, block_id=1)
        c1 = x

        x = self._depthwise_conv_block_v1(x, 128, alpha, depth_multiplier,
                                          strides=(2, 2), block_id=2)
        x = self._depthwise_conv_block_v1(x, 128, alpha, depth_multiplier, block_id=3)
        c2 = x

        x = self._depthwise_conv_block_v1(x, 256, alpha, depth_multiplier,
                                          strides=(2, 2), block_id=4)
        x = self._depthwise_conv_block_v1(x, 256, alpha, depth_multiplier, block_id=5)
        c3 = x

        x = self._depthwise_conv_block_v1(x, 512, alpha, depth_multiplier,
                                          strides=(2, 2), block_id=6, dilation=dilation[0])
        x = self._depthwise_conv_block_v1(x, 512, alpha, depth_multiplier, block_id=7, dilation=dilation[0])
        x = self._depthwise_conv_block_v1(x, 512, alpha, depth_multiplier, block_id=8, dilation=dilation[0])
        x = self._depthwise_conv_block_v1(x, 512, alpha, depth_multiplier, block_id=9, dilation=dilation[0])
        x = self._depthwise_conv_block_v1(x, 512, alpha, depth_multiplier, block_id=10, dilation=dilation[0])
        x = self._depthwise_conv_block_v1(x, 512, alpha, depth_multiplier, block_id=11, dilation=dilation[0])
        c4 = x

        x = self._depthwise_conv_block_v1(x, 1024, alpha, depth_multiplier,
                                          strides=(2, 2), block_id=12, dilation=dilation[1])
        x = self._depthwise_conv_block_v1(x, 1024, alpha, depth_multiplier, block_id=13, dilation=dilation[1])
        c5 = x

        return c1, c2, c3, c4, c5

    def _mobilenet_v2(self, inputs, alpha=1.0):
        """
        call for MobileNetV2.
        :param inputs: a 4-D tensor.
        :param alpha: controls the width of the network.
        :return: the output of different stages.
        """
        dilation = self.dilation
        channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

        first_block_filters = _make_divisible(32 * alpha, 8)
        x = layers.ZeroPadding2D(padding=_correct_pad(backend, inputs, 3),
                                 name='Conv1_pad')(inputs)
        x = layers.Conv2D(first_block_filters,
                          kernel_size=3,
                          strides=(2, 2),
                          padding='valid',
                          use_bias=False,
                          name='Conv1')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name='bn_Conv1')(x)
        x = layers.ReLU(6., name='Conv1_relu')(x)

        x = self._inverted_res_block_v2(x, filters=16, alpha=alpha, stride=1,
                                        expansion=1, block_id=0)
        c1 = x

        x = self._inverted_res_block_v2(x, filters=24, alpha=alpha, stride=2,
                                        expansion=6, block_id=1)
        x = self._inverted_res_block_v2(x, filters=24, alpha=alpha, stride=1,
                                        expansion=6, block_id=2)
        c2 = x

        x = self._inverted_res_block_v2(x, filters=32, alpha=alpha, stride=2,
                                        expansion=6, block_id=3)
        x = self._inverted_res_block_v2(x, filters=32, alpha=alpha, stride=1,
                                        expansion=6, block_id=4)
        x = self._inverted_res_block_v2(x, filters=32, alpha=alpha, stride=1,
                                        expansion=6, block_id=5)
        c3 = x

        x = self._inverted_res_block_v2(x, filters=64, alpha=alpha, stride=2,
                                        expansion=6, block_id=6, dilation=dilation[0])
        x = self._inverted_res_block_v2(x, filters=64, alpha=alpha, stride=1,
                                        expansion=6, block_id=7, dilation=dilation[0])
        x = self._inverted_res_block_v2(x, filters=64, alpha=alpha, stride=1,
                                        expansion=6, block_id=8, dilation=dilation[0])
        x = self._inverted_res_block_v2(x, filters=64, alpha=alpha, stride=1,
                                        expansion=6, block_id=9, dilation=dilation[0])

        x = self._inverted_res_block_v2(x, filters=96, alpha=alpha, stride=1,
                                        expansion=6, block_id=10, dilation=dilation[0])
        x = self._inverted_res_block_v2(x, filters=96, alpha=alpha, stride=1,
                                        expansion=6, block_id=11, dilation=dilation[0])
        x = self._inverted_res_block_v2(x, filters=96, alpha=alpha, stride=1,
                                        expansion=6, block_id=12, dilation=dilation[0])
        c4 = x

        x = self._inverted_res_block_v2(x, filters=160, alpha=alpha, stride=2,
                                        expansion=6, block_id=13, dilation=dilation[1])
        x = self._inverted_res_block_v2(x, filters=160, alpha=alpha, stride=1,
                                        expansion=6, block_id=14, dilation=dilation[1])
        x = self._inverted_res_block_v2(x, filters=160, alpha=alpha, stride=1,
                                        expansion=6, block_id=15, dilation=dilation[1])

        x = self._inverted_res_block_v2(x, filters=320, alpha=alpha, stride=1,
                                        expansion=6, block_id=16, dilation=dilation[1])

        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if alpha > 1.0:
            last_block_filters = _make_divisible(1280 * alpha, 8)
        else:
            last_block_filters = 1280

        x = layers.Conv2D(last_block_filters,
                          kernel_size=1,
                          use_bias=False,
                          name='Conv_1')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name='Conv_1_bn')(x)
        x = layers.ReLU(6., name='out_relu')(x)
        c5 = x

        return c1, c2, c3, c4, c5
