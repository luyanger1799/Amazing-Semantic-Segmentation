"""
The implementation of SegNet and Bayesian-SegNet based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from models import Network
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
backend = tf.keras.backend


class SegNet(Network):
    def __init__(self, num_classes, version='SegNet', base_model='VGG16', **kwargs):
        """
        The initialization of SegNet or Bayesian-SegNet.
        :param num_classes: the number of predicted classes.
        :param version: 'SegNet' or 'Bayesian-SegNet'.
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        base_model = 'VGG16' if base_model is None else base_model
        assert version in ['SegNet', 'Bayesian-SegNet']
        assert base_model in ['VGG16',
                              'VGG19',
                              'ResNet50',
                              'ResNet101',
                              'ResNet152',
                              'DenseNet121',
                              'DenseNet169',
                              'DenseNet201',
                              'DenseNet269',
                              'MobileNetV1',
                              'MobileNetV2',
                              'Xception',
                              'Xception-DeepLab']
        super(SegNet, self).__init__(num_classes, version, base_model, **kwargs)

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size + (3,))
        return self._segnet(inputs)

    def _conv_bn_relu(self, x, filters, kernel_size=1, strides=1):
        x = layers.Conv2D(filters, kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def _segnet(self, inputs):
        num_classes = self.num_classes
        dropout = True if self.version == 'Bayesian-SegNet' else False

        x = self.encoder(inputs)

        if dropout:
            x = layers.Dropout(rate=0.5)(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = self._conv_bn_relu(x, 512, 3, strides=1)
        x = self._conv_bn_relu(x, 512, 3, strides=1)
        x = self._conv_bn_relu(x, 512, 3, strides=1)

        if dropout:
            x = layers.Dropout(rate=0.5)(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = self._conv_bn_relu(x, 512, 3, strides=1)
        x = self._conv_bn_relu(x, 512, 3, strides=1)
        x = self._conv_bn_relu(x, 256, 3, strides=1)

        if dropout:
            x = layers.Dropout(rate=0.5)(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = self._conv_bn_relu(x, 256, 3, strides=1)
        x = self._conv_bn_relu(x, 256, 3, strides=1)
        x = self._conv_bn_relu(x, 128, 3, strides=1)

        if dropout:
            x = layers.Dropout(rate=0.5)(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = self._conv_bn_relu(x, 128, 3, strides=1)
        x = self._conv_bn_relu(x, 64, 3, strides=1)

        if dropout:
            x = layers.Dropout(rate=0.5)(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = self._conv_bn_relu(x, 64, 3, strides=1)
        x = layers.Conv2D(num_classes, 1,
                          strides=1,
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        outputs = x
        return models.Model(inputs, outputs)
