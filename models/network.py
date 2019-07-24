"""
The implementation of Network based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from base_models import *


class Network(object):
    def __init__(self, num_classes, version='PAN', base_model='ResNet50', dilation=None, **kwargs):
        super(Network, self).__init__(**kwargs)
        if base_model in ['VGG16', 'VGG19']:
            self.encoder = VGG(base_model, dilation=dilation)
        elif base_model in ['ResNet50', 'ResNet101', 'ResNet152']:
            self.encoder = ResNet(base_model, dilation=dilation)
        elif base_model in ['DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet264']:
            self.encoder = DenseNet(base_model, dilation=dilation)
        elif base_model in ['Xception', 'Xception-DeepLab']:
            self.encoder = Xception(base_model, dilation=dilation)
        elif base_model in ['MobileNetV1', 'MobileNetV2']:
            self.encoder = MobileNet(base_model, dilation=dilation)
        else:
            raise ValueError('The base model {model} is not in the '
                             'supported model list!!!'.format(model=base_model))

        self.num_classes = num_classes
        self.version = version
        self.base_model = base_model

    def __call__(self, inputs, **kwargs):
        return inputs

    def get_version(self):
        return self.version

    def get_base_model(self):
        return self.base_model
