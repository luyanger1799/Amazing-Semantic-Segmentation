"""
The implementation of DenseASPP based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from utils import layers as custom_layers
from models import Network
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
backend = tf.keras.backend


class DenseASPP(Network):
    def __init__(self, num_classes, version='DenseASPP', base_model='DenseNet121', **kwargs):
        """
        The initialization of DenseASPP based.
        :param num_classes: the number of predicted classes.
        :param version: 'DenseASPP'
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        dilation = [2, 4]
        base_model = 'DenseNet121' if base_model is None else base_model

        assert version == 'DenseASPP'
        assert base_model in ['VGG16',
                              'VGG19',
                              'ResNet50',
                              'ResNet101',
                              'ResNet152',
                              'DenseNet121',
                              'DenseNet169',
                              'DenseNet201',
                              'DenseNet264',
                              'MobileNetV1',
                              'MobileNetV2',
                              'Xception-DeepLab']
        super(DenseASPP, self).__init__(num_classes, version, base_model, dilation, **kwargs)

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size + (3,))
        return self._denseaspp(inputs)

    def _dilated_conv_block(self, inputs, filters, kernel_size=3, rate=1):
        x = layers.BatchNormalization()(inputs)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size,
                          padding='same',
                          dilation_rate=rate,
                          kernel_initializer='he_normal')(x)
        return x

    def _denseaspp(self, inputs):
        _, inputs_h, inputs_w, _ = backend.int_shape(inputs)
        aspp_size = inputs_h // 8, inputs_w // 8
        num_classes = self.num_classes

        c5 = self.encoder(inputs, output_stages='c5')

        # First block rate=3
        d3 = self._dilated_conv_block(c5, 256, 1)
        d3 = self._dilated_conv_block(d3, 64, 3, rate=3)

        # Second block rate=6
        d4 = custom_layers.Concatenate(out_size=aspp_size)([c5, d3])
        d4 = self._dilated_conv_block(d4, 256, 1)
        d4 = self._dilated_conv_block(d4, 64, 3, rate=6)

        # Third block rate=12
        d5 = custom_layers.Concatenate(out_size=aspp_size)([c5, d3, d4])
        d5 = self._dilated_conv_block(d5, 256, 1)
        d5 = self._dilated_conv_block(d5, 64, 3, rate=12)

        # Forth block rate=18
        d6 = custom_layers.Concatenate(out_size=aspp_size)([c5, d3, d4, d5])
        d6 = self._dilated_conv_block(d6, 256, 1)
        d6 = self._dilated_conv_block(d6, 64, 3, rate=18)

        # Fifth block rate=24
        d7 = custom_layers.Concatenate(out_size=aspp_size)([c5, d3, d4, d5, d6])
        d7 = self._dilated_conv_block(d7, 256, 1)
        d7 = self._dilated_conv_block(d7, 64, 3, rate=24)

        x = custom_layers.Concatenate(out_size=aspp_size)([c5, d3, d4, d5, d6, d7])
        x = layers.Conv2D(num_classes, 1, strides=1, kernel_initializer='he_normal')(x)
        x = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(x)

        outputs = x
        return models.Model(inputs, outputs)
