"""
Pathology Image MetaClass for Whole Slide Breast Histopathology Images/Masks.
Provides functionality for pathology image/mask meta functions; read/show.

##################
License: Apache License 2.0
author: Caner Mercan, 2018
"""

import os
import matplotlib.pyplot as plt
import digiPath.dataNames as DN
from digiPath.utils.pim_reader import PIMRead

########################################################
### Pathology Image (.tif) and Pathology Mask (.mat) ###
########################################################

class PImage():
    def __init__(self, pim=None, path='', exist=False):
        self.pim   = pim
        self.path  = path
        self.exist = exist
    def setPath(self, path):
        self.path = path
        self.checkPath() # always update if the path exists after path is updated.
    def checkPath(self):
        self.exist = os.path.exists(self.path)
    def read(self):
        self.pim = PIMRead.readPImage(self.path)
    def readpatch(self, offRows, offCols, numRows, numCols):
        return PIMRead.readPImagepatch(self.path, offRows, offCols, numRows, numCols)
    def show(self):
        plt.imshow(self.pim)

class PMask():
    def __init__(self, pim=None, path='', exist=False):
        self.pmask = pim
        self.path  = path
        self.exist = exist
    def setPath(self, path):
        self.path = path
        self.checkPath()  
    def checkPath(self):
        self.exist = os.path.exists(self.path)
    def read(self):
        self.pmask = PIMRead.readPMask(self.path, DN.FGMASK_KEY)
    def readpatch(self, offRows, offCols, numRows, numCols):
        return PIMRead.readPMaskpatch(self.path, DN.FGMASK_KEY, offRows, offCols, numRows, numCols)
    def show(self):
        plt.imshow(self.pmask)

