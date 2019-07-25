# Amazing-Semantic-Segmentation
Semantic Segmentation on Tensorflow && keras (include FCN, UNet, SegNet, PSPNet, PAN, RefineNet, DeepLabV3, DeepLabV3+, DenseASPP, BiSegNet)
***
## Models
The project supports these semantic segmentation models as follows:
```buildoutcfg
1. FCN-8s/16s/32s
2. UNet
3. SegNet/Bayesian-SegNet
4. PSPNet
5. RefineNet
6. PAN
7. DeepLabV3
8. DeepLabV3+
9. BiSegNet

```
***
## Base Model
The project supports these backbone models as follows, and your can choose suitable base model according to your needs.
```buildoutcfg
1. VGG16/19
2. ResNet50/101/152
3. DenseNet121/169/201/264
4. MobileNetV1/V2
5. Xception/Xception-DeepLab

```
***
## Dataset Setting
The folds of your dataset must satisfy the following structures:
```buildoutcfg
- dataset
   -train
      -images
      -labels
   -valid
      -images
      -labels
   -test
      -images
      -labels

```
***
## Installation
- Numpy `pip install numpy`
- Pillow `pip install pillow`
- OpenCV `pip install opencv-python`
- Tensorflow `pip install tensorflow-gpu` version >= 1.14
***
## Usage
**Training:** 
The project contains complete codes for training, testing and predicting. 
And you can perform a simple command as this to build a model on your dataset:
```buildoutcfg
python train.py --model FCN-8s --base_model ResNet50 --dataset "dataset_path" --num_classes "num_classes"
```
The detailed command line parameters are as follows:
```buildoutcfg
usage: train.py [-h] --model MODEL [--base_model BASE_MODEL] --dataset DATASET
                --num_classes NUM_CLASSES [--random_crop RANDOM_CROP]
                [--crop_height CROP_HEIGHT] [--crop_width CROP_WIDTH]
                [--batch_size BATCH_SIZE]
                [--valid_batch_size VALID_BATCH_SIZE]
                [--num_epochs NUM_EPOCHS] [--initial_epoch INITIAL_EPOCH]
                [--h_flip H_FLIP] [--v_flip V_FLIP]
                [--brightness BRIGHTNESS [BRIGHTNESS ...]]
                [--rotation ROTATION]
                [--zoom_range ZOOM_RANGE [ZOOM_RANGE ...]]
                [--channel_shift CHANNEL_SHIFT]
                [--data_aug_rate DATA_AUG_RATE]
                [--checkpoint_freq CHECKPOINT_FREQ]
                [--validation_freq VALIDATION_FREQ]
                [--num_valid_images NUM_VALID_IMAGES]
                [--data_shuffle DATA_SHUFFLE] [--random_seed RANDOM_SEED]
                [--weights WEIGHTS]

```
```buildoutcfg
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Choose the semantic segmentation methods.
  --base_model BASE_MODEL
                        Choose the backbone model.
  --dataset DATASET     The path of the dataset.
  --num_classes NUM_CLASSES
                        The number of classes to be segmented.
  --random_crop RANDOM_CROP
                        Whether to randomly crop the image.
  --crop_height CROP_HEIGHT
                        The height to crop the image.
  --crop_width CROP_WIDTH
                        The width to crop the image.
  --batch_size BATCH_SIZE
                        The training batch size.
  --valid_batch_size VALID_BATCH_SIZE
                        The validation batch size.
  --num_epochs NUM_EPOCHS
                        The number of epochs to train for.
  --initial_epoch INITIAL_EPOCH
                        The initial epoch of training.
  --h_flip H_FLIP       Whether to randomly flip the image horizontally.
  --v_flip V_FLIP       Whether to randomly flip the image vertically.
  --brightness BRIGHTNESS [BRIGHTNESS ...]
                        Randomly change the brightness (list).
  --rotation ROTATION   The angle to randomly rotate the image.
  --zoom_range ZOOM_RANGE [ZOOM_RANGE ...]
                        The times for zooming the image.
  --channel_shift CHANNEL_SHIFT
                        The channel shift range.
  --data_aug_rate DATA_AUG_RATE
                        The rate of data augmentation.
  --checkpoint_freq CHECKPOINT_FREQ
                        How often to save a checkpoint.
  --validation_freq VALIDATION_FREQ
                        How often to perform validation.
  --num_valid_images NUM_VALID_IMAGES
                        The number of images used for validation.
  --data_shuffle DATA_SHUFFLE
                        Whether to shuffle the data.
  --random_seed RANDOM_SEED
                        The random shuffle seed.
  --weights WEIGHTS     The path of weights to be loaded.

```
If you only want to use the model with your own training code, you can do as this:
```buildoutcfg
from builders.model_builder import builder

model, base_model = builder(num_classes, input_size, model='SegNet', base_model=None)

```
**Note:** If you don't give the parameter "base_model", the default backbone will be used.

**Testing:** 
Similarly, you can evaluate the model on your own dataset:
```buildoutcfg
python test.py --model FCN-8s --base_model ResNet50 --dataset "dataset_path" --num_classes "num_classes" --weights "weights_path"
```
**Note:** If the parameter "weights" is None, the weigths saved in default path will be loaded.  

**Predicting:** 
You can get the prediction of a single RGB image as this:
```buildoutcfg
python predict.py --model FCN-8s --base_model ResNet50 --num_classes "num_classes" --weights "weights_path" --image_path "image_path"
``` 
***
## PyPI
The installation of this project from PyPI will be supported later.
***
## Others
If you like this work, please give me a star! And if you find
any errors or have any suggestions, please contact me.
***
## Contact
**GitHub:** `luyanger1799`\
**Email:** `luyanger1799@outlook.com`
