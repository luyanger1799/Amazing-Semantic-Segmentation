"""
The file defines the training process.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from utils.data_generator import ImageDataGenerator
from utils.helpers import get_dataset_info, check_related_path
from utils.optimizers import *
from utils.losses import *
from utils.learning_rate import *
from utils.metrics import MeanIoU
from utils import utils
from builders import builder
import tensorflow as tf
import argparse
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
parser.add_argument('--dataset', help='The path of the dataset.', type=str, default='CamVid')
parser.add_argument('--loss', help='The loss function for traing.', type=str, default=None,
                    choices=['ce', 'focal_loss', 'miou_loss', 'self_balanced_focal_loss'])
parser.add_argument('--num_classes', help='The number of classes to be segmented.', type=int, default=32)
parser.add_argument('--random_crop', help='Whether to randomly crop the image.', type=str2bool, default=False)
parser.add_argument('--crop_height', help='The height to crop the image.', type=int, default=256)
parser.add_argument('--crop_width', help='The width to crop the image.', type=int, default=256)
parser.add_argument('--batch_size', help='The training batch size.', type=int, default=5)
parser.add_argument('--valid_batch_size', help='The validation batch size.', type=int, default=1)
parser.add_argument('--num_epochs', help='The number of epochs to train for.', type=int, default=100)
parser.add_argument('--initial_epoch', help='The initial epoch of training.', type=int, default=0)
parser.add_argument('--h_flip', help='Whether to randomly flip the image horizontally.', type=str2bool, default=False)
parser.add_argument('--v_flip', help='Whether to randomly flip the image vertically.', type=str2bool, default=False)
parser.add_argument('--brightness', help='Randomly change the brightness (list).', type=float, default=None, nargs='+')
parser.add_argument('--rotation', help='The angle to randomly rotate the image.', type=float, default=0.)
parser.add_argument('--zoom_range', help='The times for zooming the image.', type=float, default=0., nargs='+')
parser.add_argument('--channel_shift', help='The channel shift range.', type=float, default=0.)
parser.add_argument('--data_aug_rate', help='The rate of data augmentation.', type=float, default=0.)
parser.add_argument('--checkpoint_freq', help='How often to save a checkpoint.', type=int, default=1)
parser.add_argument('--validation_freq', help='How often to perform validation.', type=int, default=1)
parser.add_argument('--num_valid_images', help='The number of images used for validation.', type=int, default=20)
parser.add_argument('--data_shuffle', help='Whether to shuffle the data.', type=str2bool, default=True)
parser.add_argument('--random_seed', help='The random shuffle seed.', type=int, default=None)
parser.add_argument('--weights', help='The path of weights to be loaded.', type=str, default=None)
parser.add_argument('--steps_per_epoch', help='The training steps of each epoch', type=int, default=None)
parser.add_argument('--lr_scheduler', help='The strategy to schedule learning rate.', type=str, default='cosine_decay',
                    choices=['step_decay', 'poly_decay', 'cosine_decay'])
parser.add_argument('--warmup_steps', help='The steps for lr warm up.', type=int, default=5)
parser.add_argument('--learning_rate', help='The initial learning rate.', type=float, default=3e-4)
parser.add_argument('--optimizer', help='The optimizer for training.', type=str, default='adam',
                    choices=['sgd', 'adam', 'nadam', 'adamw', 'nadamw', 'sgdw'])

args = parser.parse_args()

# check related paths
paths = check_related_path(os.getcwd())

# get image and label file names for training and validation
train_image_names, train_label_names, valid_image_names, valid_label_names, _, _ = get_dataset_info(args.dataset)

# build the model
net, base_model = builder(args.num_classes, (args.crop_height, args.crop_width), args.model, args.base_model)

# summary
net.summary()

# load weights
if args.weights is not None:
    print('Loading the weights...')
    net.load_weights(args.weights)

# chose loss
losses = {'ce': categorical_crossentropy_with_logits,
          'focal_loss': focal_loss(),
          'miou_loss': miou_loss(num_classes=args.num_classes),
          'self_balanced_focal_loss': self_balanced_focal_loss()}
loss = losses[args.loss] if args.loss is not None else categorical_crossentropy_with_logits

# chose optimizer
total_iterations = len(train_image_names) * args.num_epochs // args.batch_size
wd_dict = utils.get_weight_decays(net)
ordered_values = []
weight_decays = utils.fill_dict_in_order(wd_dict, ordered_values)

optimizers = {'adam': tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
              'nadam': tf.keras.optimizers.Nadam(learning_rate=args.learning_rate),
              'sgd': tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.99),
              'adamw': AdamW(learning_rate=args.learning_rate, batch_size=args.batch_size,
                             total_iterations=total_iterations),
              'nadamw': NadamW(learning_rate=args.learning_rate, batch_size=args.batch_size,
                               total_iterations=total_iterations),
              'sgdw': SGDW(learning_rate=args.learning_rate, momentum=0.99, batch_size=args.batch_size,
                           total_iterations=total_iterations)}

# lr schedule strategy
lr_decays = {'step_decay': step_decay(args.learning_rate, args.num_epochs, args.warmup_steps),
             'poly_decay': poly_decay(args.learning_rate, args.num_epochs, args.warmup_steps),
             'cosine_decay': cosine_decay(args.num_epochs, args.learning_rate, warmup_steps=args.warmup_steps)}
lr_decay = lr_decays[args.lr_scheduler]

# compile the model
net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=loss,
            metrics=[MeanIoU(args.num_classes)])
# data generator
# data augmentation setting
train_gen = ImageDataGenerator(random_crop=args.random_crop,
                               rotation_range=args.rotation,
                               brightness_range=args.brightness,
                               zoom_range=args.zoom_range,
                               channel_shift_range=args.channel_shift,
                               horizontal_flip=args.v_flip,
                               vertical_flip=args.v_flip)

valid_gen = ImageDataGenerator()

train_generator = train_gen.flow(images_list=train_image_names,
                                 labels_list=train_label_names,
                                 num_classes=args.num_classes,
                                 batch_size=args.batch_size,
                                 target_size=(args.crop_height, args.crop_width),
                                 shuffle=args.data_shuffle,
                                 seed=args.random_seed,
                                 data_aug_rate=args.data_aug_rate)

valid_generator = valid_gen.flow(images_list=valid_image_names,
                                 labels_list=valid_label_names,
                                 num_classes=args.num_classes,
                                 batch_size=args.valid_batch_size,
                                 target_size=(args.crop_height, args.crop_width))

# callbacks setting
# checkpoint setting
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(paths['checkpoints_path'],
                          '{model}_based_on_{base}_'.format(model=args.model, base=base_model) +
                          'miou_{val_mean_io_u:04f}_' + 'ep_{epoch:02d}.h5'),
    save_best_only=True, period=args.checkpoint_freq, monitor='val_mean_io_u', mode='max')
# tensorboard setting
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=paths['logs_path'])
# learning rate scheduler setting
learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decay)

callbacks = [model_checkpoint, tensorboard, learning_rate_scheduler]

# begin training
print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Num Images -->", len(train_image_names))
print("Model -->", args.model)
print("Base Model -->", base_model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Initial Epoch -->", args.initial_epoch)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", args.num_classes)

print("Data Augmentation:")
print("\tData Augmentation Rate -->", args.data_aug_rate)
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("\tZoom -->", args.zoom_range)
print("\tChannel Shift -->", args.channel_shift)

print("")

# some other training parameters
steps_per_epoch = len(train_image_names) // args.batch_size if not args.steps_per_epoch else args.steps_per_epoch
validation_steps = args.num_valid_images // args.valid_batch_size

# training...
net.fit_generator(train_generator,
                  steps_per_epoch=steps_per_epoch,
                  epochs=args.num_epochs,
                  callbacks=callbacks,
                  validation_data=valid_generator,
                  validation_steps=validation_steps,
                  validation_freq=args.validation_freq,
                  max_queue_size=10,
                  workers=os.cpu_count(),
                  use_multiprocessing=True,
                  initial_epoch=args.initial_epoch)

# save weights
net.save(filepath=os.path.join(
    paths['weights_path'], '{model}_based_on_{base_model}.h5'.format(model=args.model, base_model=base_model)))
