"""
Adopted and adjusted from the pytorch tutorials
author: Caner Mercan
"""

import os 
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
#from torchvision import datasets, models, transforms

class ConvNet():

    def __init__(self, pre_trained=True, num_classes=4):
        self.num_classes = num_classes
        ### load the model
        self.model = torchvision.models.alexnet(pretrained=pre_trained)
        self.__set_model(pre_trained)

    def __set_model(self, pre_trained):
        if pre_trained:
            ### make only the last layer parameters trainable (Weights + Bias)
            for idx, param in enumerate(self.model.parameters()):
                if idx < len(list(self.model.parameters()))-2:
                    param.requires_grad = False
                    
            ### change the last layer's output size as the number of classes        
            self.model.classifier[-1].out_features = self.num_classes
        else:
            ### add a final layer with the number of neurons == number of classes.
            module_name = str(len(self.model.classifier))
            out_features = self.model.classifier[-1].out_features
            self.model.classifier.add_module(module_name, torch.nn.Linear(out_features, self.num_classes))

    def get_model(self):
        return copy.deepcopy(self.model)

    def fit(self, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
        """
        training process for the 
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        since = time.time()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
