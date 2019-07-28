"""
The implementation of Data Generator based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from tensorflow.python.keras.preprocessing.image import Iterator
from keras_applications import imagenet_utils
from utils.utils import *
import tensorflow as tf
import numpy as np

keras_utils = tf.keras.utils


class DataIterator(Iterator):
    def __init__(self,
                 image_data_generator,
                 images_list,
                 labels_list,
                 num_classes,
                 batch_size,
                 target_size,
                 shuffle=True,
                 seed=None,
                 data_aug_rate=0.):
        num_images = len(images_list)

        self.image_data_generator = image_data_generator
        self.images_list = images_list
        self.labels_list = labels_list
        self.num_classes = num_classes
        self.target_size = target_size
        self.data_aug_rate = data_aug_rate

        super(DataIterator, self).__init__(num_images, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(shape=(len(index_array),) + self.target_size + (3,))
        batch_y = np.zeros(shape=(len(index_array),) + self.target_size + (self.num_classes,))

        for i, idx in enumerate(index_array):
            image, label = load_image(self.images_list[idx]), load_image(self.labels_list[idx])
            # random crop
            if self.image_data_generator.random_crop:
                image, label = random_crop(image, label, self.target_size)
            else:
                image, label = resize_image(image, label, self.target_size)
            # data augmentation
            if np.random.uniform(0., 1.) < self.data_aug_rate:
                # random vertical flip
                if np.random.randint(2):
                    image, label = random_vertical_flip(image, label, self.image_data_generator.vertical_flip)
                # random horizontal flip
                if np.random.randint(2):
                    image, label = random_horizontal_flip(image, label, self.image_data_generator.horizontal_flip)
                # random brightness
                if np.random.randint(2):
                    image, label = random_brightness(image, label, self.image_data_generator.brightness_range)
                # random rotation
                if np.random.randint(2):
                    image, label = random_rotation(image, label, self.image_data_generator.rotation_range)
                # random channel shift
                if np.random.randint(2):
                    image, label = random_channel_shift(image, label, self.image_data_generator.channel_shift_range)
                # random zoom
                if np.random.randint(2):
                    image, label = random_zoom(image, label, self.image_data_generator.zoom_range)

            image = imagenet_utils.preprocess_input(image.astype('float32'), data_format='channels_last',
                                                    mode='torch')
            label = one_hot(label, self.num_classes)

            batch_x[i], batch_y[i] = image, label

        return batch_x, batch_y


class ImageDataGenerator(object):
    def __init__(self,
                 random_crop=False,
                 rotation_range=0,
                 brightness_range=None,
                 zoom_range=0.0,
                 channel_shift_range=0.0,
                 horizontal_flip=False,
                 vertical_flip=False):
        self.random_crop = random_crop
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def flow(self,
             images_list,
             labels_list,
             num_classes,
             batch_size,
             target_size,
             shuffle=True,
             seed=None,
             data_aug_rate=0.):
        return DataIterator(image_data_generator=self,
                            images_list=images_list,
                            labels_list=labels_list,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            target_size=target_size,
                            shuffle=shuffle,
                            seed=seed,
                            data_aug_rate=data_aug_rate)
