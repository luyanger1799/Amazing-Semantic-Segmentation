# Amazing-Semantic-Segmentation
[![python](https://img.shields.io/badge/Python-3.x-ff69b4.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)
[![tensorflow](https://img.shields.io/badge/Tensorflow-1.1x%7C2.0-brightgreen.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)
[![OpenCV](https://img.shields.io/badge/OpenCV-3.x%7C4.x-orange.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)
[![Apache](https://img.shields.io/badge/Apache-2.0-blue.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)

>Amazing Semantic Segmentation on Tensorflow && keras (include FCN, UNet, SegNet, PSPNet, PAN, RefineNet, DeepLabV3, DeepLabV3+, DenseASPP, BiSegNet ...)
***
## Models
The project supports these semantic segmentation models as follows:

>1. FCN-8s/16s/32s - [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)
>2. UNet - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
>3. SegNet - [SegNet:A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf)
>4. Bayesian-SegNet - [Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding](https://arxiv.org/pdf/1511.02680v2.pdf)
>5. PSPNet - [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf)
>6. RefineNet - [RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/pdf/1611.06612.pdf)
>7. PAN - [Pyramid Attention Network for Semantic Segmentation](https://arxiv.org/pdf/1805.10180.pdf)
>8. DeepLabV3 - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf)
>9. DeepLabV3Plus - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
>10. DenseASPP - [DenseASPP for Semantic Segmentation in Street Scenes](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
>11. BiSegNet - [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/pdf/1808.00897.pdf)


***
## Base Models
The project supports these backbone models as follows, and your can choose suitable base model according to your needs.

>1. VGG16/19 - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
>2. ResNet50/101/152 - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
>3. DenseNet121/169/201/264 - [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
>4. MobileNetV1 - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
>5. MobileNetV2 - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)
>6. Xception - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)
>7. Xception-DeepLab - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)



***
## Dataset Setting
The folds of your dataset must satisfy the following structures:
```buildoutcfg
|-- dataset
|  |-- train
|  |  |-- images
|  |  |-- labels
|  |-- valid
|  |  |-- images
|  |  |-- labels
|  |-- test
|  |  |-- images
|  |  |-- labels
```
***
## Installation
- Numpy `pip install numpy`
- Pillow `pip install pillow`
- OpenCV `pip install opencv-python`
- Tensorflow `pip install tensorflow-gpu`  

**Note:** The recommended version of tensorflow-gpu is 1.14 or 2.0. And if your tensorflow version is lower, you 
need to modify some API or upgrade your tensorflow. 
***
## Usage
**Download:**
You can download the project through this command:  
`git clone git@github.com:luyanger1799/Amazing-Semantic-Segmentation.git`  

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
If you only want to use the model in your own training code, you can do as this:
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
Alternatively, you can install the project through PyPI.
```
pip install semantic-segmentation
```
And you can use model_builders to build different models or directly call the class of semantic segmentation.
```
from semantic_segmentation import model_builders
net, base_net = model_builders(num_classes, input_size, model='SegNet', base_model=None)
```
or
```
from semantic_segmentation import models
net = models.FCN(num_classes, version='FCN-8s')(input_size=input_size)
```
***
## Pre-trained
Due to my limited computing resources, there is no pre-training model yet. And maybe it will be added in the future.
***
## Feedback
If you like this work, please give me a star! And if you find
any errors or have any suggestions, please contact me.  

**GitHub:** `luyanger1799`\
**Email:** `luyanger1799@outlook.com` 
