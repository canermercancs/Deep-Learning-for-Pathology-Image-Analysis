
import torchvision


MEAN_CONVNET = {
	'alexnet'	: [0.485, 0.456, 0.406],
	'vgg' 		: [0.485, 0.456, 0.406]
	} # mean for alexnet images
STD_CONVNET  = {
	'alexnet'	: [0.229, 0.224, 0.225],
	'vgg' 		: [0.229, 0.224, 0.225]
	} # standard deviation for alexnet images

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
