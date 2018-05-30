
import torchvision


MEAN_CONVNET = {'alexnet': [0.485, 0.456, 0.406]} # mean for alexnet images
STD_CONVNET  = {'alexnet': [0.229, 0.224, 0.225]} # standard deviation for alexnet images

VGG_TYPES = {
	'vgg11' : torchvision.models.vgg11
	'vgg13' : torchvision.models.vgg13
	'vgg16' : torchvision.models.vgg16
	'vgg19' : torchvision.models.vgg19
	'vgg11bn' : torchvision.models.vgg11bn
	'vgg13bn' : torchvision.models.vgg13bn
	'vgg16bn' : torchvision.models.vgg16bn
	'vgg19bn' : torchvision.models.vgg19bn
}