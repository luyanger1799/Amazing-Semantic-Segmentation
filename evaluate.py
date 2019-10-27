"""
The file defines the evaluate process on target dataset.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from sklearn.metrics import multilabel_confusion_matrix
from utils.helpers import *
from utils.utils import load_image
import numpy as np
import argparse
import sys
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
parser.add_argument('--dataset', help='The path of the dataset.', type=str, default='CamVid')
parser.add_argument('--crop_height', help='The height to crop the image.', type=int, default=256)
parser.add_argument('--crop_width', help='The width to crop the image.', type=int, default=256)
parser.add_argument('--predictions', help='The path of predicted image.', type=str, required=True)

args = parser.parse_args()

# check related paths
paths = check_related_path(os.getcwd())

# get image and label file names for training and validation
_, _, _, _, _, test_label_names = get_dataset_info(args.dataset)

# get color info
csv_file = os.path.join(args.dataset, 'class_dict.csv')

class_names, _ = get_colored_info(csv_file)

# get the prediction file name list
if not os.path.exists(args.predictions):
    raise ValueError('the path of predictions does not exit.')

prediction_names = []
for file in sorted(os.listdir(args.predictions)):
    prediction_names.append(os.path.join(args.predictions, file))

# evaluated classes
evaluated_classes = get_evaluated_classes(os.path.join(args.dataset, 'evaluated_classes.txt'))

num_classes = len(class_names)
class_iou = dict()
for name in evaluated_classes:
    class_iou[name] = list()

class_idx = dict(zip(class_names, range(num_classes)))

# begin evaluate
assert len(test_label_names) == len(prediction_names)

for i, (name1, name2) in enumerate(zip(test_label_names, prediction_names)):
    sys.stdout.write('\rRunning test image %d / %d' % (i + 1, len(test_label_names)))
    sys.stdout.flush()

    label = np.array(cv2.resize(load_image(name1),
                                dsize=(args.crop_width, args.crop_height), interpolation=cv2.INTER_NEAREST))
    pred = np.array(cv2.resize(load_image(name2),
                               dsize=(args.crop_width, args.crop_height), interpolation=cv2.INTER_NEAREST))

    confusion_matrix = multilabel_confusion_matrix(label.flatten(), pred.flatten(), labels=list(class_idx.values()))
    for eval_cls in evaluated_classes:
        eval_idx = class_idx[eval_cls]
        (tn, fp), (fn, tp) = confusion_matrix[eval_idx]

        if tp + fn > 0:
            class_iou[eval_cls].append(tp / (tp + fp + fn))

print('\n****************************************')
print('* The IoU of each class is as follows: *')
print('****************************************')
for eval_cls in evaluated_classes:
    class_iou[eval_cls] = np.mean(class_iou[eval_cls])
    print('{cls:}: {iou:.4f}'.format(cls=eval_cls, iou=class_iou[eval_cls]))

print('\n**********************************************')
print('* The Mean IoU of all classes is as follows: *')
print('**********************************************')
print('Mean IoU: {mean_iou:.4f}'.format(mean_iou=np.mean(list(class_iou.values()))))
