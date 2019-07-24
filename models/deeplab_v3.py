"""
The implementation of DeepLabV3 based on Tensorflow.

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


class DeepLabV3(Network):
    def __init__(self, num_classes, version='DeepLabV3', base_model='ResNet50', **kwargs):
        """
        The initialization of DeepLabV3.
        :param num_classes: the number of predicted classes.
        :param version: 'DeepLabV3'
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        dilation = [1, 2]
        base_model = 'ResNet50' if base_model is None else base_model

        assert version == 'DeepLabV3'
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
        super(DeepLabV3, self).__init__(num_classes, version, base_model, dilation, **kwargs)
        self.dilation = dilation

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size + (3,))
        return self._deeplabv3(inputs)

    def _deeplabv3(self, inputs):
        multi_grid = [1, 2, 4]
        num_classes = self.num_classes
        dilation = self.dilation

        _, h, w, _ = backend.int_shape(inputs)
        self.aspp_size = (h // 16, w // 16)

        x = self.encoder(inputs, output_stages='c4')

        x = self._conv_block(x, 3, [512, 512, 2048], stage=5, block='a', dilation=dilation[1])
        for i in range(2):
            x = self._identity_block(x, 3, [512, 512, 2048],
                                     stage=5,
                                     block=chr(ord('b') + i),
                                     dilation=dilation[1] * multi_grid[i])
        x = self._aspp(x, 256)
        x = layers.Conv2D(num_classes, 1, strides=1, kernel_initializer='he_normal')(x)
        x = layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(x)

        outputs = x
        return models.Model(inputs, outputs)

    def _aspp(self, x, out_filters):
        xs = list()
        x1 = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(x)
        xs.append(x1)

        for i in range(3):
            xi = layers.Conv2D(out_filters, 3,
                               strides=1,
                               padding='same',
                               dilation_rate=6 * (i + 1))(x)
            xs.append(xi)
        img_pool = custom_layers.GlobalAveragePooling2D(keep_dims=True)(x)
        img_pool = layers.Conv2D(out_filters, 1, 1, kernel_initializer='he_normal')(img_pool)
        img_pool = layers.UpSampling2D(size=self.aspp_size, interpolation='bilinear')(img_pool)
        xs.append(img_pool)

        x = custom_layers.Concatenate(out_size=self.aspp_size)(xs)
        x = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        return x

    def _identity_block(self, input_tensor, kernel_size, filters, stage, block, dilation=1):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b',
                          dilation_rate=dilation)(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def _conv_block(self,
                    input_tensor,
                    kernel_size,
                    filters,
                    stage,
                    block,
                    strides=(2, 2),
                    dilation=1):
        """A block that has a conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        strides = (1, 1) if dilation > 1 else strides

        x = layers.Conv2D(filters1, (1, 1),
                          strides=strides,
                          name=conv_name_base + '2a',
                          kernel_initializer='he_normal')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size,
                          padding='same',
                          name=conv_name_base + '2b',
                          kernel_initializer='he_normal',
                          dilation_rate=dilation)(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          name=conv_name_base + '2c',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = layers.Conv2D(filters3, (1, 1),
                                 strides=strides,
                                 name=conv_name_base + '1',
                                 kernel_initializer='he_normal')(input_tensor)
        shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
