"""
The implementation of DeepLabV3Plus based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from utils import layers as my_layers
from models import Network
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
backend = tf.keras.backend


class DeepLabV3Plus(Network):
    def __init__(self, num_classes, version='DeepLabV3Plus', base_model='Xception-DeepLab', **kwargs):
        """
        The initialization of DeepLabV3Plus.
        :param num_classes: the number of predicted classes.
        :param version: 'DeepLabV3Plus'
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        dilation = [1, 2]
        base_model = 'Xception-DeepLab' if base_model is None else base_model

        assert version == 'DeepLabV3Plus'
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
        super(DeepLabV3Plus, self).__init__(num_classes, version, base_model, dilation, **kwargs)
        self.dilation = dilation

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size + (3,))
        return self._deeplab_v3_plus(inputs)

    def _deeplab_v3_plus(self, inputs):
        num_classes = self.num_classes
        _, h, w, _ = backend.int_shape(inputs)
        self.aspp_size = (h // 16, w // 16)

        if self.base_model in ['VGG16',
                               'VGG19',
                               'ResNet50',
                               'ResNet101',
                               'ResNet152',
                               'MobileNetV1',
                               'MobileNetV2']:
            c2, c5 = self.encoder(inputs, output_stages=['c2', 'c5'])
        else:
            c2, c5 = self.encoder(inputs, output_stages=['c1', 'c5'])

        x = self._aspp(c5, 256)
        x = layers.Dropout(rate=0.5)(x)

        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        x = self._conv_bn_relu(x, 48, 1, strides=1)

        x = my_layers.Concatenate(out_size=self.aspp_size)([x, c2])
        x = self._conv_bn_relu(x, 256, 3, 1)
        x = layers.Dropout(rate=0.5)(x)

        x = self._conv_bn_relu(x, 256, 3, 1)
        x = layers.Dropout(rate=0.1)(x)

        x = layers.Conv2D(num_classes, 1, strides=1)(x)
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

        outputs = x
        return models.Model(inputs, outputs)

    def _conv_bn_relu(self, x, filters, kernel_size, strides=1):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def _aspp(self, x, out_filters):
        xs = list()
        x1 = layers.Conv2D(out_filters, 1, strides=1)(x)
        xs.append(x1)

        for i in range(3):
            xi = layers.Conv2D(out_filters, 3, strides=1, padding='same', dilation_rate=6 * (i + 1))(x)
            xs.append(xi)
        img_pool = my_layers.GlobalAveragePooling2D(keep_dims=True)(x)
        img_pool = layers.Conv2D(out_filters, 1, 1, kernel_initializer='he_normal')(img_pool)
        img_pool = layers.UpSampling2D(size=self.aspp_size, interpolation='bilinear')(img_pool)
        xs.append(img_pool)

        x = my_layers.Concatenate(out_size=self.aspp_size)(xs)
        x = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        return x
