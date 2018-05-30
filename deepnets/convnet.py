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
from .models.convmodel import ConvnetModel
from .models.convprops as CP
#from torchvision import datasets, models, transforms



class Convnet():

    def __init__(self, model_name = '', pre_trained = True, num_classes = 4, model_dir = ''):
        
    	# model name cannot be empty
    	assert model_name.rstrip().lstrip()

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
            trainedNet = torchvision.models.alexnet(pretrained = self.model_pretrained)
        elif self.model_name.startswith('vgg'):
        	vgg_model  = CP.VGG_TYPE[self.model_name] 
        	trainedNet = vgg_model(pretrained = self.model_pretrained)
        else:
            raise NotImplementedError
		self.model      = ConvnetModel(self.num_classes, self.model_name, trainedNet)
        # set where the model will run on.        
        self.model.to(self.processor)
        print(f'Running on {self.processor}')

#        if torch.cuda.is_available():
#            self.model = self.model.cuda()
#            print('Running on GPU...')
#        else:
#            print('Running on CPU...')            
        #self.model.features = torch.nn.DataParallel(self.model.features)
    def __assert_model_props__(self):
        assert self.model is not None
        assert os.path.exists(self.model_dir)        

    def get_model(self):
        return self.model
    def get_params(self):
        return self.model.parameters()


    # @todo: move to convmodel.ConvnetModel class.
    def save_model(self, fname):
        fpath = os.path.join(self.model_dir, fname)    
        #pdb.set_trace()  
        fpath = update_path(fpath)  
        torch.save(self.model.state_dict(), fpath)

    # @todo: move to convmodel.ConvnetModel class.    
    def load_model(self, epoch=None):
        fname = f'finetuned({self.model_name})model'
        if epoch is not None:        
            fname += f'_epoch({epoch})'
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, fname)))

    def fit(self, dataloaders, criterion, optimizer, scheduler, num_epochs=(0, 500)):  
        #pdb.set_trace()
        epoch_stats   = Stats(phases = ['train', 'val'])
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

        #best_model_W  = copy.deepcopy(self.model.state_dict())
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
                    inputs = inputs.to(self.processor)
                    labels = labels.to(self.processor)
                
                    #inputs, labels = Variable(inputs), Variable(labels)			
                    #if torch.cuda.is_available():
                    #    inputs, labels = inputs.cuda(), labels.cuda()    

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

                    ## forward
                    #outputs = self.model(inputs)
                    #loss = criterion(outputs, labels)
                    #_, preds = torch.max(outputs, 1)

                    ## backward + optimize only if in training phase
                    #if phase == 'train':
                    #    loss.backward()
                    #    optimizer.step()

                    # statistics
                    # print statistics
                    # running_loss += loss.data[0] # old notation.
                    running_loss += loss.item()
                    if i % 50 == 49:    # print every 50 minibatches
                        print('[%d, %d] loss: %.3f' % (epoch+1, i+1, running_loss/50))
                        running_loss = 0.0				

                    # running_loss_epoch    += loss.data[0] * inputs.size(0) # old notation.
                    running_loss_epoch      += loss.item() * inputs.size(0)
                    running_corrects_epoch  += torch.sum(preds == labels).item()
                
                epoch_loss = running_loss_epoch / dataset_sizes[phase]
                epoch_acc  = running_corrects_epoch / dataset_sizes[phase]
                print(f'{self.model_name} at {self.model_dir}')
                print('Epoch #{} {} Loss: {:.3f}'.format(epoch+1, phase, epoch_loss))
                print('Epoch #{} {} Accuracy: {:.3f}'.format(epoch+1, phase, epoch_acc))

                # keeping track of the epoch progress
                epoch_stats(phase, epoch_loss, epoch_acc)
                epoch_stats.write2file(self.model_dir)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    #best_model_W = copy.deepcopy(model.state_dict())
                    best_acc = epoch_acc
                    fname    = f'finetuned({self.model_name})bestmodel'
                    self.save_model(fname) 
	
    		# write model to file at every 50 epochs.
            if epoch % 50 == 49:
                fname = f'finetuned({self.model_name})model_epoch({epoch+1})'
                self.save_model(fname)
                
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

