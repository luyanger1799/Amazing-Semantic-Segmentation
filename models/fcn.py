"""
The implementation of FCN-8s/16s/32s based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from models import Network
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
backend = tf.keras.backend


class FCN(Network):
    def __init__(self, num_classes, version='FCN-8s', base_model='VGG16', **kwargs):
        """
        The initialization of FCN-8s/16s/32s.
        :param num_classes: the number of predicted classes.
        :param version: 'FCN-8s', 'FCN-16s' or 'FCN-32s'.
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        fcn = {'FCN-8s': self._fcn_8s,
               'FCN-16s': self._fcn_16s,
               'FCN-32s': self._fcn_32s}
        base_model = 'VGG16' if base_model is None else base_model

        assert version in fcn
        self.fcn = fcn[version]
        super(FCN, self).__init__(num_classes, version, base_model, **kwargs)

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size + (3,))
        return self.fcn(inputs)

    def _conv_relu(self, x, filters, kernel_size=1):
        x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = layers.ReLU()(x)
        return x

    def _fcn_32s(self, inputs):
        num_classes = self.num_classes

        x = self.encoder(inputs)
        x = self._conv_relu(x, 4096, 7)
        x = layers.Dropout(rate=0.5)(x)
        x = self._conv_relu(x, 4096, 1)
        x = layers.Dropout(rate=0.5)(x)

        x = layers.Conv2D(num_classes, 1, kernel_initializer='he_normal')(x)
        x = layers.Conv2DTranspose(num_classes, 64, strides=32, padding='same', kernel_initializer='he_normal')(x)

        outputs = x
        return models.Model(inputs, outputs, name=self.version)

    def _fcn_16s(self, inputs):
        num_classes = self.num_classes

        if self.base_model in ['DenseNet121',
                               'DenseNet169',
                               'DenseNet201',
                               'DenseNet264',
                               'Xception',
                               'Xception-DeepLab']:
            c4, c5 = self.encoder(inputs, output_stages=['c3', 'c5'])
        else:
            c4, c5 = self.encoder(inputs, output_stages=['c4', 'c5'])

        x = self._conv_relu(c5, 4096, 7)
        x = layers.Dropout(rate=0.5)(x)
        x = self._conv_relu(x, 4096, 1)
        x = layers.Dropout(rate=0.5)(x)

        x = layers.Conv2D(num_classes, 1, kernel_initializer='he_normal')(x)
        x = layers.Conv2DTranspose(num_classes, 4,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer='he_normal')(x)
        c4 = layers.Conv2D(num_classes, 1, kernel_initializer='he_normal')(c4)
        x = layers.Add()([x, c4])

        x = layers.Conv2DTranspose(num_classes, 32,
                                   strides=16,
                                   padding='same',
                                   kernel_initializer='he_normal')(x)

        outputs = x
        return models.Model(inputs, outputs, name=self.version)

    def _fcn_8s(self, inputs):
        num_classes = self.num_classes

        if self.base_model in ['VGG16',
                               'VGG19',
                               'ResNet50',
                               'ResNet101',
                               'ResNet152',
                               'MobileNetV1',
                               'MobileNetV2']:
            c3, c4, c5 = self.encoder(inputs, output_stages=['c3', 'c4', 'c5'])
        else:
            c3, c4, c5 = self.encoder(inputs, output_stages=['c2', 'c3', 'c5'])

        x = self._conv_relu(c5, 4096, 7)
        x = layers.Dropout(rate=0.5)(x)
        x = self._conv_relu(x, 4096, 1)
        x = layers.Dropout(rate=0.5)(x)

        x = layers.Conv2D(num_classes, 1, kernel_initializer='he_normal')(x)
        x = layers.Conv2DTranspose(num_classes, 4,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer='he_normal')(x)
        c4 = layers.Conv2D(num_classes, 1)(c4)
        x = layers.Add()([x, c4])

        x = layers.Conv2DTranspose(num_classes, 4,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer='he_normal')(x)
        c3 = layers.Conv2D(num_classes, 1)(c3)
        x = layers.Add()([x, c3])

        x = layers.Conv2DTranspose(num_classes, 16,
                                   strides=8,
                                   padding='same',
                                   kernel_initializer='he_normal')(x)

        outputs = x
        return models.Model(inputs, outputs, name=self.version)
