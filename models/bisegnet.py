"""
The implementation of BiSegNet based on Tensorflow.

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


class BiSegNet(Network):
    def __init__(self, num_classes, version='BiSegNet', base_model='Xception', **kwargs):
        """
        The initialization of BiSegNet.
        :param num_classes: the number of predicted classes.
        :param version: 'BiSegNet'
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        base_model = 'Xception' if base_model is None else base_model

        assert version == 'BiSegNet'
        super(BiSegNet, self).__init__(num_classes, version, base_model, **kwargs)

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size+(3,))
        return self._bisegnet(inputs)

    def _conv_block(self, x, filters, kernel_size=3, strides=1):
        x = layers.Conv2D(filters, kernel_size, strides, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def _attention_refinement_module(self, x):
        # Global average pooling
        _, _, _, c = backend.int_shape(x)

        glb = my_layers.GlobalAveragePooling2D(keep_dims=True)(x)
        glb = layers.Conv2D(c, 1, strides=1, kernel_initializer='he_normal')(glb)
        glb = layers.BatchNormalization()(glb)
        glb = layers.Activation(activation='sigmoid')(glb)

        x = layers.Multiply()([x, glb])

        return x

    def _feature_fusion_module(self, input_1, input_2, filters):
        inputs = layers.Concatenate()([input_1, input_2])
        inputs = self._conv_block(inputs, filters=filters, kernel_size=3)

        # Global average pooling
        _, _, _, c = backend.int_shape(inputs)

        glb = my_layers.GlobalAveragePooling2D(keep_dims=True)(inputs)
        glb = layers.Conv2D(filters, 1, strides=1, activation='relu', kernel_initializer='he_normal')(glb)
        glb = layers.Conv2D(filters, 1, strides=1, activation='sigmoid', kernel_initializer='he_normal')(glb)

        x = layers.Multiply()([inputs, glb])

        return x

    def _bisegnet(self, inputs):
        num_classes = self.num_classes

        # the spatial path
        sx = self._conv_block(inputs, 64, 3, 2)
        sx = self._conv_block(sx, 128, 3, 2)
        sx = self._conv_block(sx, 256, 3, 2)

        # the context path
        if self.base_model in ['VGG16',
                               'VGG19',
                               'ResNet50',
                               'ResNet101',
                               'ResNet152',
                               'MobileNetV1',
                               'MobileNetV2',
                               'Xception',
                               'Xception-DeepLab']:
            c4, c5 = self.encoder(inputs, output_stages=['c4', 'c5'])
        else:
            c4, c5 = self.encoder(inputs, output_stages=['c3', 'c5'])

        c4 = self._attention_refinement_module(c4)
        c5 = self._attention_refinement_module(c5)

        glb = my_layers.GlobalAveragePooling2D(keep_dims=True)(c5)
        c5 = layers.Multiply()([c5, glb])

        # combining the paths
        c4 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(c4)
        c5 = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(c5)

        cx = layers.Concatenate()([c4, c5])

        x = self._feature_fusion_module(sx, cx, num_classes)

        x = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(x)
        x = layers.Conv2D(num_classes, 1, 1, kernel_initializer='he_normal')(x)

        outputs = x

        return models.Model(inputs, outputs)
