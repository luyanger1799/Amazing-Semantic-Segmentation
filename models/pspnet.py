"""
The implementation of PSPNet based on Tensorflow.

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


class PSPNet(Network):
    def __init__(self, num_classes, version='PSPNet', base_model='ResNet50', **kwargs):
        """
        The initialization of PSPNet.
        :param num_classes: the number of predicted classes.
        :param version: 'PSPNet'
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        dilation = [2, 4]
        base_model = 'ResNet50' if base_model is None else base_model

        assert version == 'PSPNet'
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
        super(PSPNet, self).__init__(num_classes, version, base_model, dilation, **kwargs)

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size + (3,))
        return self._pspnet(inputs)

    def _pspnet(self, inputs):
        num_classes = self.num_classes
        _, inputs_h, inputs_w, _ = backend.int_shape(inputs)

        h, w = inputs_h // 8, inputs_w // 8
        x = self.encoder(inputs)

        if not (h % 6 == 0 and w % 6 == 0):
            raise ValueError('\'pyramid pooling\' size must be divided by 6, but received {size}'.format(size=(h, w)))
        pool_size = [(h, w),
                     (h // 2, w // 2),
                     (h // 3, w // 3),
                     (h // 6, w // 6)]

        # pyramid pooling
        x1 = custom_layers.GlobalAveragePooling2D(keep_dims=True)(x)
        x1 = layers.Conv2D(512, 1, strides=1, kernel_initializer='he_normal')(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.ReLU()(x1)
        x1 = layers.UpSampling2D(size=pool_size[0])(x1)

        x2 = layers.AveragePooling2D(pool_size=pool_size[1])(x)
        x2 = layers.Conv2D(512, 1, strides=1, kernel_initializer='he_normal')(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.ReLU()(x2)
        x2 = layers.UpSampling2D(size=pool_size[1])(x2)

        x3 = layers.AveragePooling2D(pool_size=pool_size[2])(x)
        x3 = layers.Conv2D(512, 1, strides=1, kernel_initializer='he_normal')(x3)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.ReLU()(x3)
        x3 = layers.UpSampling2D(size=pool_size[2])(x3)

        x6 = layers.AveragePooling2D(pool_size=pool_size[3])(x)
        x6 = layers.Conv2D(512, 1, strides=1, kernel_initializer='he_normal')(x6)
        x6 = layers.BatchNormalization()(x6)
        x6 = layers.ReLU()(x6)
        x6 = layers.UpSampling2D(size=pool_size[3])(x6)

        x = layers.Concatenate()([x, x1, x2, x3, x6])

        x = layers.Conv2D(512, 3, strides=1, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(num_classes, 1, strides=1, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(x)

        outputs = x

        return models.Model(inputs, outputs)
