"""
The implementation of some layers based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf

layers = tf.keras.layers
backend = tf.keras.backend


class GlobalAveragePooling2D(layers.GlobalAveragePooling2D):
    def __init__(self, keep_dims=False, **kwargs):
        super(GlobalAveragePooling2D, self).__init__(**kwargs)
        self.keep_dims = keep_dims

    def call(self, inputs):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).call(inputs)
        else:
            return backend.mean(inputs, axis=[1, 2], keepdims=True)

    def compute_output_shape(self, input_shape):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).compute_output_shape(input_shape)
        else:
            input_shape = tf.TensorShape(input_shape).as_list()
            return tf.TensorShape([input_shape[0], 1, 1, input_shape[3]])

    def get_config(self):
        config = super(GlobalAveragePooling2D, self).get_config()
        config['keep_dim'] = self.keep_dims
        return config


class Concatenate(layers.Concatenate):
    def __init__(self, out_size=None, axis=-1, name=None):
        super(Concatenate, self).__init__(axis=axis, name=name)
        self.out_size = out_size

    def call(self, inputs):
        return backend.concatenate(inputs, self.axis)

    def build(self, input_shape):
        pass

    def compute_output_shape(self, input_shape):
        if self.out_size is None:
            return super(Concatenate, self).compute_output_shape(input_shape)
        else:
            if not isinstance(input_shape, list):
                raise ValueError('A `Concatenate` layer should be called '
                                 'on a list of inputs.')
            input_shapes = input_shape
            output_shape = list(input_shapes[0])
            for shape in input_shapes[1:]:
                if output_shape[self.axis] is None or shape[self.axis] is None:
                    output_shape[self.axis] = None
                    break
                output_shape[self.axis] += shape[self.axis]
            return tuple([output_shape[0]] + list(self.out_size) + [output_shape[-1]])

    def get_config(self):
        config = super(Concatenate, self).get_config()
        config['out_size'] = self.out_size
        return config


class PixelShuffle(layers.Layer):
    def __init__(self, block_size=2, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        if isinstance(block_size, int):
            self.block_size = block_size
        elif isinstance(block_size, (list, tuple)):
            assert len(block_size) == 2 and block_size[0] == block_size[1]
            self.block_size = block_size[0]
        else:
            raise ValueError('error \'block_size\'.')

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        return tf.nn.depth_to_space(inputs, self.block_size)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()

        _, h, w, c = input_shape

        new_h = h * self.block_size
        new_w = w * self.block_size
        new_c = c / self.block_size ** 2

        return tf.TensorShape([input_shape[0], new_h, new_w, new_c])

    def get_config(self):
        config = super(PixelShuffle, self).get_config()
        config['block_size'] = self.block_size
        return config
