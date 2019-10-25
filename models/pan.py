"""
The implementation of PAN (Pyramid Attention Networks) based on Tensorflow.

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


class PAN(Network):
    def __init__(self, num_classes, version='PAN', base_model='ResNet50', **kwargs):
        """
        The initialization of PAN.
        :param num_classes: the number of predicted classes.
        :param version: 'PAN'
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        base_model = 'ResNet50' if base_model is None else base_model
        assert version == 'PAN'

        dilation = [2, 4]
        if base_model in ['VGG16',
                          'VGG19',
                          'MobileNetV1',
                          'MobileNetV2',
                          'ResNet50',
                          'ResNet101',
                          'ResNet152']:
            self.up_size = [(1, 1), (1, 1), (2, 2), (4, 4)]
        elif base_model in ['DenseNet121',
                            'DenseNet169',
                            'DenseNet201',
                            'DenseNet264',
                            'Xception-DeepLab']:
            self.up_size = [(1, 1), (1, 1), (1, 1), (8, 8)]
        else:
            raise ValueError('The base model \'{model}\' is not '
                             'supported in PAN.'.format(model=base_model))

        super(PAN, self).__init__(num_classes, version, base_model, dilation, **kwargs)

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size + (3,))
        return self._pan(inputs)

    def _conv_bn_relu(self, x, filters, kernel_size, strides=1):
        x = layers.Conv2D(filters, kernel_size, strides, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def _fpa(self, x, out_filters):
        _, h, w, _ = backend.int_shape(x)

        # global average pooling
        glb = custom_layers.GlobalAveragePooling2D(keep_dims=True)(x)
        glb = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(glb)

        # down
        down1 = layers.AveragePooling2D(pool_size=(2, 2))(x)
        down1 = self._conv_bn_relu(down1, 1, 7, 1)

        down2 = layers.AveragePooling2D(pool_size=(2, 2))(down1)
        down2 = self._conv_bn_relu(down2, 1, 5, 1)

        down3 = layers.AveragePooling2D(pool_size=(2, 2))(down2)
        down3 = self._conv_bn_relu(down3, 1, 3, 1)

        down1 = self._conv_bn_relu(down1, 1, 7, 1)
        down2 = self._conv_bn_relu(down2, 1, 5, 1)
        down3 = self._conv_bn_relu(down3, 1, 3, 1)

        # up
        up2 = layers.UpSampling2D(size=(2, 2))(down3)
        up2 = layers.Add()([up2, down2])

        up1 = layers.UpSampling2D(size=(2, 2))(up2)
        up1 = layers.Add()([up1, down1])

        up = layers.UpSampling2D(size=(2, 2))(up1)

        x = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        # multiply
        x = layers.Multiply()([x, up])

        # add
        x = layers.Add()([x, glb])

        return x

    def _gau(self, x, y, out_filters, up_size=(2, 2)):
        glb = custom_layers.GlobalAveragePooling2D(keep_dims=True)(y)
        glb = layers.Conv2D(out_filters, 1, strides=1, activation='sigmoid', kernel_initializer='he_normal')(glb)

        x = self._conv_bn_relu(x, out_filters, 3, 1)
        x = layers.Multiply()([x, glb])

        y = layers.UpSampling2D(size=up_size, interpolation='bilinear')(y)

        y = layers.Add()([x, y])

        return y

    def _pan(self, inputs):
        num_classes = self.num_classes
        up_size = self.up_size

        c2, c3, c4, c5 = self.encoder(inputs, output_stages=['c2', 'c3', 'c4', 'c5'])

        y = self._fpa(c5, num_classes)

        y = self._gau(c4, y, num_classes, up_size[0])
        y = self._gau(c3, y, num_classes, up_size[1])
        y = self._gau(c2, y, num_classes, up_size[2])

        y = layers.UpSampling2D(size=up_size[3], interpolation='bilinear')(y)

        outputs = y

        return models.Model(inputs, outputs, name=self.version)
