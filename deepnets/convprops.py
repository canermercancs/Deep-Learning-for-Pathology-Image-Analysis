"""
author: Caner Mercan
"""
import torchvision

MODULE_NAMES = 'alexnet', 'vgg'
MEAN_CONVNET = {
    MODULE_NAMES[0] : [0.485, 0.456, 0.406],
    MODULE_NAMES[1] : [0.485, 0.456, 0.406]
    } # mean for alexnet images
STD_CONVNET  = {
    MODULE_NAMES[0] : [0.229, 0.224, 0.225],
    MODULE_NAMES[1] : [0.229, 0.224, 0.225]
    } # standard deviation for alexnet images
# efficient because doesn't load functions into memory; only their handles.
VGG_MODELS = {
    'vgg11' : torchvision.models.vgg11,
    'vgg13' : torchvision.models.vgg13,
    'vgg16' : torchvision.models.vgg16,
    'vgg19' : torchvision.models.vgg19,
    'vgg11_bn' : torchvision.models.vgg11_bn,
    'vgg13_bn' : torchvision.models.vgg13_bn,
    'vgg16_bn' : torchvision.models.vgg16_bn,
    'vgg19_bn' : torchvision.models.vgg19_bn
}
# number of 'classifier' modules to pop for feature extraction.
FEATURE_POP_OPS = {
    MODULE_NAMES[0] : 2,
    MODULE_NAMES[1] : 3
}