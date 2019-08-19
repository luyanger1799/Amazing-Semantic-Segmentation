"""
The file defines the predict process of a single RGB image.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from keras_applications import imagenet_utils
from utils.utils import load_image, decode_one_hot
from utils.helpers import check_related_path
from builders import builder
from PIL import Image
import numpy as np
import argparse
import cv2
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Choose the semantic segmentation methods.', type=str, required=True)
parser.add_argument('--base_model', help='Choose the backbone model.', type=str, default=None)
parser.add_argument('--num_classes', help='The number of classes to be segmented.', type=int, required=True)
parser.add_argument('--crop_height', help='The height to crop the image.', type=int, default=256)
parser.add_argument('--crop_width', help='The width to crop the image.', type=int, default=256)
parser.add_argument('--weights', help='The path of weights to be loaded.', type=str, default=None)
parser.add_argument('--image_path', help='The path of predicted image.', type=str, required=True)

args = parser.parse_args()

# check related paths
paths = check_related_path(os.getcwd())

# check the image path
if not os.path.exists(args.image_path):
    raise ValueError('The path \'{image_path}\' does not exist the image file.'.format(image_path=args.image_path))

# build the model
net, base_model = builder(args.num_classes, (args.crop_height, args.crop_width), args.model, args.base_model)

# load weights
print('Loading the weights...')
if args.weights is None:
    net.load_weights(filepath=os.path.join(
        paths['weigths_path'], '{model}_based_on_{base_model}.h5'.format(model=args.model, base_model=base_model)))
else:
    if not os.path.exists(args.weights):
        raise ValueError('The weights file does not exist in \'{path}\''.format(path=args.weights))
    net.load_weights(args.weights)

# begin testing
print("\n***** Begin testing *****")
print("Model -->", args.model)
print("Base Model -->", base_model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", args.num_classes)

print("")

# load_image
image = cv2.resize(load_image(args.image_path),
                   dsize=(args.crop_width, args.crop_height),
                   interpolation=cv2.INTER_LINEAR)
image = imagenet_utils.preprocess_input(image, data_format='channels_last', mode='tf')

# image processing
if np.ndim(image) == 3:
    image = np.expand_dims(image, axis=0)
assert np.ndim(image) == 4

# get the prediction
prediction = net.predict_on_batch(image)

if np.ndim(prediction) == 4:
    prediction = np.squeeze(prediction, axis=0)

# decode one-hot
prediction = decode_one_hot(prediction)

# get PIL file
prediction = Image.fromarray(np.uint8(prediction))

# save the prediction
_, file_name = os.path.split(args.image_path)
prediction.save(os.path.join(paths['prediction_path'], file_name))
