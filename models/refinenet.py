"""
The implementation of RefineNet based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from models import Network
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
backend = tf.keras.backend


class RefineNet(Network):
    def __init__(self, num_classes, version='RefineNet', base_model='ResNet50', **kwargs):
        """
        The initialization of RefineNet.
        :param num_classes: the number of predicted classes.
        :param version: 'RefineNet'
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        base_model = 'ResNet50' if base_model is None else base_model

        assert version == 'RefineNet'
        assert base_model in ['VGG16',
                              'VGG19',
                              'ResNet50',
                              'ResNet101',
                              'ResNet152',
                              'MobileNetV1',
                              'MobileNetV2']
        super(RefineNet, self).__init__(num_classes, version, base_model, **kwargs)

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size + (3,))
        return self._refinenet(inputs)

    def _refinenet(self, inputs):
        num_classes = self.num_classes

        xs = self.encoder(inputs, output_stages=['c2', 'c3', 'c4', 'c5'])[::-1]

        g = [None, None, None, None]
        h = [None, None, None, None]

        for i in range(4):
            h[i] = layers.Conv2D(256, 1, strides=1, kernel_initializer='he_normal')(xs[i])

        g[0] = self._refine_block(high_inputs=None, low_inputs=h[0])
        g[1] = self._refine_block(g[0], h[1])
        g[2] = self._refine_block(g[1], h[2])
        g[3] = self._refine_block(g[2], h[3])

        x = layers.Conv2D(num_classes, 1, strides=1, kernel_initializer='he_normal')(g[3])
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

        outputs = x

        return models.Model(inputs, outputs, name=self.version)

    def _residual_conv_unit(self, inputs, features=256, kernel_size=3):
        x = layers.ReLU()(inputs)
        x = layers.Conv2D(features, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(features, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = layers.Add()([inputs, x])
        return x

    def _chained_residual_pooling(self, inputs, features=256):
        x_relu = layers.ReLU()(inputs)
        x = layers.MaxPool2D((5, 5), strides=1, padding='same')(x_relu)
        x = layers.Conv2D(features, 3, padding='same', kernel_initializer='he_normal')(x)
        x_sum_1 = layers.Add()([x, x_relu])

        x = layers.MaxPool2D((5, 5), strides=1, padding='same')(x_relu)
        x = layers.Conv2D(features, 3, padding='same', kernel_initializer='he_normal')(x)
        x_sum_2 = layers.Add()([x, x_sum_1])

        return x_sum_2

    def _multi_resolution_fusion(self, high_inputs=None, low_inputs=None, features=256):

        if high_inputs is None:  # refineNet block 4
            rcu_low_1 = low_inputs[0]
            rcu_low_2 = low_inputs[1]

            rcu_low_1 = layers.Conv2D(features, 3, padding='same', kernel_initializer='he_normal')(rcu_low_1)
            rcu_low_2 = layers.Conv2D(features, 3, padding='same', kernel_initializer='he_normal')(rcu_low_2)

            return layers.Add()([rcu_low_1, rcu_low_2])

        else:
            rcu_low_1 = low_inputs[0]
            rcu_low_2 = low_inputs[1]

            rcu_low_1 = layers.Conv2D(features, 3, padding='same', kernel_initializer='he_normal')(rcu_low_1)
            rcu_low_2 = layers.Conv2D(features, 3, padding='same', kernel_initializer='he_normal')(rcu_low_2)

            rcu_low = layers.Add()([rcu_low_1, rcu_low_2])

            rcu_high_1 = high_inputs[0]
            rcu_high_2 = high_inputs[1]

            rcu_high_1 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(
                layers.Conv2D(features, 3, padding='same', kernel_initializer='he_normal')(rcu_high_1))
            rcu_high_2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(
                layers.Conv2D(features, 3, padding='same', kernel_initializer='he_normal')(rcu_high_2))

            rcu_high = layers.Add()([rcu_high_1, rcu_high_2])

            return layers.Add()([rcu_low, rcu_high])

    def _refine_block(self, high_inputs=None, low_inputs=None):

        if high_inputs is None:  # block 4
            rcu_low_1 = self._residual_conv_unit(low_inputs, features=256)
            rcu_low_2 = self._residual_conv_unit(low_inputs, features=256)
            rcu_low = [rcu_low_1, rcu_low_2]

            fuse = self._multi_resolution_fusion(high_inputs=None, low_inputs=rcu_low, features=256)
            fuse_pooling = self._chained_residual_pooling(fuse, features=256)
            output = self._residual_conv_unit(fuse_pooling, features=256)
            return output
        else:
            rcu_low_1 = self._residual_conv_unit(low_inputs, features=256)
            rcu_low_2 = self._residual_conv_unit(low_inputs, features=256)
            rcu_low = [rcu_low_1, rcu_low_2]

            rcu_high_1 = self._residual_conv_unit(high_inputs, features=256)
            rcu_high_2 = self._residual_conv_unit(high_inputs, features=256)
            rcu_high = [rcu_high_1, rcu_high_2]

            fuse = self._multi_resolution_fusion(rcu_high, rcu_low, features=256)
            fuse_pooling = self._chained_residual_pooling(fuse, features=256)
            output = self._residual_conv_unit(fuse_pooling, features=256)
            return output
