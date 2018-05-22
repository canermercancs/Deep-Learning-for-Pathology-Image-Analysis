"""
author: Caner Mercan
"""

from __future__ import print_function, division

import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import deepnets.convprops as PR
# import matplotlib.pyplot as plt

class Convdata():
    def __init__(self, model_name='alexnet', data_path=''):
        self.loaders 	 = None
        self.class_names = []
        self.data_path   = data_path
        self.MEAN_GLOBAL = PR.MEAN_CONVNET[model_name]
        self.STD_GLOBAL  = PR.STD_CONVNET[model_name]     

    def __call__(self, batch_size = 64):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN_GLOBAL, self.STD_GLOBAL)
            ]), 
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN_GLOBAL, self.STD_GLOBAL)
            ])
        }
        image_datasets 	= {x: datasets.ImageFolder(os.path.join(self.data_path, x),
                                                	data_transforms[x]) 
							for x in ['train', 'val']}
        dataloaders 	= {x: torch.utils.data.DataLoader(image_datasets[x], 
														batch_size = batch_size, 
														shuffle = True, 
														num_workers = 8) 
							for x in ['train', 'val']}

        self.loaders 		= dataloaders 
        self.class_names 	= {x: image_datasets[x].classes for x in ['train', 'val']}

    def getData(self):
        return self.loaders
	
    def getImages(self, datatype = 'train'):
        """Imshow for Tensor."""
        inputs, classes = next(iter(self.loaders[datatype]))
        title = [self.class_names[datatype][x] for x in classes]
        inp = torchvision.utils.make_grid(inputs)
        inp = inp.numpy().transpose((1, 2, 0))

        mean = np.array(self.MEAN_GLOBAL)
        std = np.array(self.STD_GLOBAL)
        inp = inp * std + mean
        out = np.clip(inp, 0, 1)
        return out, title

