"""
The implementation of some helpers.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import numpy as np
import warnings
import csv
import os


def get_dataset_info(dataset_path):
    image_label_paths = check_dataset_path(dataset_path)
    image_label_names = list()

    for i, path in enumerate(image_label_paths):
        names = list()
        if path is not None:
            files = sorted(os.listdir(path))
            for file in files:
                names.append(os.path.join(path, file))
        image_label_names.append(names)

    assert len(image_label_names[0]) == len(image_label_names[1])
    assert len(image_label_names[2]) == len(image_label_names[3])

    return image_label_names


def check_dataset_path(dataset_path):
    primary_directory = ['train', 'valid', 'test']
    secondary_directory = ['images', 'labels']

    if not os.path.exists(dataset_path):
        raise ValueError('The path of the dataset does not exist.')
    else:
        train_path = os.path.join(dataset_path, primary_directory[0])
        valid_path = os.path.join(dataset_path, primary_directory[1])
        test_path = os.path.join(dataset_path, primary_directory[2])
        if not os.path.exists(train_path):
            raise ValueError('The path of the training data does not exist.')
        if not os.path.exists(valid_path):
            raise ValueError('The path of the validation data does not exist.')
        if not os.path.exists(test_path):
            warnings.warn('The path of the test data does not exist. ')

        train_image_path = os.path.join(train_path, secondary_directory[0])
        train_label_path = os.path.join(train_path, secondary_directory[1])
        if not os.path.exists(train_image_path) or not os.path.exists(train_label_path):
            raise ValueError('The path of images or labels for training does not exist.')

        valid_image_path = os.path.join(valid_path, secondary_directory[0])
        valid_label_path = os.path.join(valid_path, secondary_directory[1])
        if not os.path.exists(valid_image_path) or not os.path.exists(valid_label_path):
            raise ValueError('The path of images or labels for validation does not exist.')

        test_image_path = os.path.join(test_path, secondary_directory[0])
        test_label_path = os.path.join(test_path, secondary_directory[1])
        if not os.path.exists(test_image_path) or not os.path.exists(test_label_path):
            warnings.warn('The path of images or labels for test does not exist.')
            test_image_path = None
            test_label_path = None

        return train_image_path, train_label_path, valid_image_path, valid_label_path, test_image_path, test_label_path


def check_related_path(current_path):
    assert os.path.exists(current_path)

    checkpoints_path = os.path.join(current_path, 'checkpoints')
    logs_path = os.path.join(checkpoints_path, 'logs')
    weights_path = os.path.join(current_path, 'weights')
    prediction_path = os.path.join(current_path, 'predictions')

    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    paths = {'checkpoints_path': checkpoints_path,
             'logs_path': logs_path,
             'weights_path': weights_path,
             'prediction_path': prediction_path}
    return paths


def get_colored_info(csv_path):
    if not os.path.exists(csv_path):
        raise ValueError('The path \'{path:}\' of csv file does not exist!'.format(path=csv_path))

    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == '.csv':
        raise ValueError('File is not a CSV!')

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csv_file:
        file_reader = csv.reader(csv_file, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    return class_names, label_values


def get_evaluated_classes(file_path):
    if not os.path.exists(file_path):
        raise ValueError('The path of evaluated classes file does not exist!')

    with open(file_path, 'r') as file:
        evaluated_classes = list(map(lambda z: z.strip(), file.readlines()))

    return evaluated_classes


def color_encode(image, color_values):
    color_codes = np.array(color_values)
    x = color_codes[image.astype(int)]

    return x
