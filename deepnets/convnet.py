"""
author: Caner Mercan
"""

import pdb
import os 
import sys
import time
import copy
import numpy as np
from sklearn.metrics import confusion_matrix as confmat
import torch
import torchvision
from torch.autograd import Variable
# my own modules
from deepnets import convprops as CP
from deepnets.utils import error_handling as ERR
from deepnets.models.convmodel import ConvModel


class ConvNet():
    def __init__(self, model_name = '', pre_trained = True, num_classes = 4, model_dir = ''):
        self.model              = None
        self.model_name         = model_name
        self.model_pretrained   = pre_trained
        self.model_dir          = model_dir
        self.num_classes        = num_classes
        self.processor          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.__make_model_dir__()
        self.__set_model_props__()
        self.__assert_model_props__()

    def __make_model_dir__(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def __set_model_props__(self):
        if self.model_name.startswith('alexnet'):
            premodel = torchvision.models.alexnet(pretrained = self.model_pretrained)
        elif self.model_name.startswith('vgg'):
            try:
                assert self.model_name in CP.VGG_MODELS
            except AssertionError:
                print(ERR.CONV_MODEL_KEY_ERROR['vgg'])
            vgg_model   = CP.VGG_MODELS[self.model_name] 
            premodel    = vgg_model(pretrained = self.model_pretrained)
        else:
            raise NotImplementedError
        
        # set the model and its parameters.
        self.model = ConvModel(premodel, self.model_name, self.model_pretrained, self.num_classes )
        # set where the model will run on.        
        self.model.to(self.processor)
        # give info on processing device
        print(f'Running on {self.processor}')

    def __assert_model_props__(self):
        assert self.model is not None
        assert os.path.exists(self.model_dir)        

    def get_model(self):
        return self.model
    def get_params(self):
        return self.model.parameters()
    def save_model(self, epoch=None):
        if epoch is None:   
            fname = f'finetuned({self.model_name})model_best'            
        else:               
            fname = f'finetuned({self.model_name})model_epoch({epoch})'
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, fname))
    def load_model(self, epoch=None):
        if epoch is None:
            fname = f'finetuned({self.model_name})model_best'
        else:        
            fname = f'finetuned({self.model_name})model_epoch({epoch})'
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, fname)))

    def fit(self, dataloaders, criterion, optimizer, scheduler, num_epochs=(0, 500)):  
        #pdb.set_trace()
        epoch_stats     = Stats(phases = ['train', 'val'])
        dataset_sizes   = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
        # class_names     = dataloaders['train'].dataset.classes 

        #best_model_W  = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        for epoch in range(num_epochs[0], num_epochs[1]):
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss_epoch      = 0.0
                running_corrects_epoch  = 0.0
                running_loss            = 0.0

                # Iterate over data.
                y_preds = torch.LongTensor().to(self.processor)
                y_trues = torch.LongTensor().to(self.processor)
                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(self.processor)
                    labels = labels.to(self.processor)
                
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs     = self.model(inputs)
                        _, preds    = torch.max(outputs, 1)
                        loss        = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()
                    if i % 100 == 99:    # print every 50 minibatches
                        print('[%d, %d] loss: %.3f' % (epoch+1, i+1, running_loss/100))
                        running_loss = 0.0				

                    running_loss_epoch      += loss.item() * inputs.size(0)
                    running_corrects_epoch  += torch.sum(preds == labels).item()
                    
                    #pdb.set_trace()
                    y_preds = torch.cat((y_preds, preds), dim=0)
                    y_trues = torch.cat((y_trues, labels), dim=0)
                    
                epoch_loss = running_loss_epoch / dataset_sizes[phase]
                epoch_acc  = running_corrects_epoch / dataset_sizes[phase]
                print(f'{self.model_name} at {self.model_dir}')
                print('Epoch #{} {} Loss: {:.3f}'.format(epoch+1, phase, epoch_loss))
                print('Epoch #{} {} Accuracy: {:.3f}'.format(epoch+1, phase, epoch_acc))
                print(confmat(y_trues.cpu().numpy(), y_preds.cpu().numpy()))
                
                # keeping track of the epoch progress
                epoch_stats(phase, epoch_loss, epoch_acc)
                epoch_stats.write2file(self.model_dir)

                # deep copy the best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    self.save_model()

            # write model to file at every 50 epochs.
            if epoch % 50 == 49:
                self.save_model(epoch+1)
                
        #self.model.load_state_dict(best_model_W)
        #return self.model

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

