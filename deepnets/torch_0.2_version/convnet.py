"""
author: Caner Mercan
"""

import pdb
import os 
import time
import copy
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
# from deepnets.models.finetuned_cnnmodel import FinetunedModel
from .models.finetuned_cnnmodel import FinetunedModel
#from torchvision import datasets, models, transforms

class Convnet():

    def __init__(self, model_name = 'alexnet', pre_trained = True, num_classes = 4, model_dir = ''):
        self.model_dir      = model_dir
        self.model_name     = model_name
        self.num_classes    = num_classes

        if self.model_name.startswith('alexnet'):
            pre_model  = torchvision.models.alexnet(pretrained = pre_trained)
            self.model = FinetunedModel(num_classes, model_name, pre_model)
        else:
            raise NotImplementedError
        self.__model_dir()
        self.__set_model_props(pre_trained)

    def __model_dir(self):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def __set_model_props(self, pre_trained):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print('Running on GPU...')
        else:
            print('Running on CPU...')            
        #self.model.features = torch.nn.DataParallel(self.model.features)

    def get_model(self):
        return self.model
    def get_params(self):
        return self.model.parameters()

    def save_model(self, fname):
        fpath = os.path.join(self.model_dir, fname)    
        #pdb.set_trace()    
        fpath = update_path(fpath)
        torch.save(self.model.state_dict(), fpath)
    def load_model(self, epoch=None):
        fname = f'finetuned({self.model_name})model'
        if epoch is not None:        
            fname += f'_epoch({epoch})'
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, fname)))

    def fit(self, dataloaders, criterion, optimizer, scheduler, num_epochs=(0, 500)):  
        epoch_stats   = Stats(phases = ['train', 'val'])
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
        for epoch in range(num_epochs[0], num_epochs[1]):

            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss_epoch      = 0.0
                running_corrects_epoch  = 0.0
                running_loss = 0.0
                # Iterate over data.
                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs, labels = Variable(inputs), Variable(labels)			
                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()    

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    # print statistics
                    running_loss += loss.data[0]
                    if i % 50 == 49:    # print every 50 minibatches
                        print('[%d, %d] loss: %.3f' % (epoch+1, i+1, running_loss/50))
                        running_loss = 0.0				

                    running_loss_epoch += loss.data[0] * inputs.size(0)
                    running_corrects_epoch += torch.sum(preds == labels).data[0]
                
                epoch_loss = running_loss_epoch / dataset_sizes[phase]
                epoch_acc = running_corrects_epoch / dataset_sizes[phase]
                print(f'{self.model_name} at {self.model_dir}')
                print('Epoch #{} {} Loss: {:.3f}'.format(epoch+1, phase, epoch_loss))
                print('Epoch #{} {} Accuracy: {:.3f}'.format(epoch+1, phase, epoch_acc))

                # keeping track of the epoch progress
                epoch_stats(phase, epoch_loss, epoch_acc)
                epoch_stats.write2file(self.model_dir)
                
			# write model to file at every 50 epochs.
            if epoch % 50 == 49:
                fname = f'finetuned({self.model_name})model_epoch({epoch+1})'
                self.save_model(fname)

        return epoch_stats

import pickle
class Stats():
    def __init__(self, phases=['train', 'val']):
        self.num_epoch  = 0
        self.file_name  = 'epoch_stats'
        self.epoch_time = {p:[] for p in phases}
        self.epoch_loss = {p:[] for p in phases}    
        self.epoch_acc  = {p:[] for p in phases}
        
    def __call__(self, phase, loss, cost):
        self.epoch_loss[phase].append(loss)
        self.epoch_acc[phase].append(cost)
        self.num_epoch += 1

    def write2file(self, fdir):
        fpath = os.path.join(fdir, self.file_name)
        with open(fpath, 'wb') as f:
            fpath = update_path(fpath)
            pickle.dump([self.epoch_loss, self.epoch_acc], f)
        
    def readfile(self, fdir):
        fpath = os.path.join(fdir, self.file_name)
        with open(fpath, 'rb') as f:
            stats = pickle.load(f)
        self.epoch_loss = stats[0]
        self.epoch_acc  = stats[1]
        return stats
 
def update_path(fpath):
    aux, idx = fpath, 1
    while os.path.isfile(aux):
        aux = fpath + '_v' + str(idx)
        idx += 1
    return aux       

