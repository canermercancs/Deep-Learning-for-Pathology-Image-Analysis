"""
author: Caner Mercan
"""

from __future__ import print_function, division

import pdb
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from deepnets import convprops as CP


class ConvData():
    def __init__(self, model_name='', data_path=''):
        self.class_names = []
        self.loaders     = None
        self.data_path   = data_path
        
        # since there are many variations of vgg models, encode their keys as 'vgg' in short.
        model_name       = 'vgg' if model_name.startswith('vgg') else model_name
        self.MEAN_GLOBAL = CP.MEAN_CONVNET[model_name]
        self.STD_GLOBAL  = CP.STD_CONVNET[model_name]     

    def __call__(self, batch_size = 64):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN_GLOBAL, self.STD_GLOBAL)
            ]), 
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN_GLOBAL, self.STD_GLOBAL)
            ])
        }
        image_datasets = {phase: datasets.ImageFolder(os.path.join(self.data_path, phase),
                                                    data_transforms[phase]) 
                                                    for phase in ['train', 'val']}
        dataloaders = {phase: torch.utils.data.DataLoader(image_datasets[phase], 
                                                    batch_size = batch_size, 
                                                    shuffle = True, 
                                                    num_workers = 8) 
                                                    for phase in ['train', 'val']}
        self.loaders     = dataloaders 
        self.class_names = {phase: image_datasets[phase].classes for phase in ['train', 'val']}

    def getData(self):
        return self.loaders
    
    def getImages(self, datatype = 'train'):
        """Imshow for Tensor."""
        inputs, classes = next(iter(self.loaders[datatype]))
        title = [self.class_names[datatype][c] for c in classes]
        inp = torchvision.utils.make_grid(inputs)
        inp = inp.numpy().transpose((1, 2, 0))

        mean = np.array(self.MEAN_GLOBAL)
        std = np.array(self.STD_GLOBAL)
        inp = inp * std + mean
        out = np.clip(inp, 0, 1)
        return out, title

