"""
Pathology Image Reading for Whole Slide Breast Histopathology Images/Masks.
Provides functionality for pathology image/mask functions; read/read patch.

##################
License: Apache License 2.0
author: Caner Mercan, 2018
"""
import numpy as np
import gdal
import h5py                             # to read mat v7.3 files.

##################################################################
### Generic .tif Image(&patch) and .mat file(&partial) Readers ###
##################################################################

class PIMRead():
    """
    Reading pathology images from .tif image files
    Reading pathology masks from .mat data files
    """
    @staticmethod
    def readPImage(pimPath):
        return PIMRead.readPImagepatch(pimPath)        
    @staticmethod
    def readPImagepatch(pimPath, offsetRows=0, offsetCols=0, numRows=-1, numCols=-1):    
        pim = None
        try:
            gdalObj = gdal.Open(pimPath)
            if numRows == -1 and numCols == -1: # read the whole image
                pim = gdalObj.ReadAsArray()                        
            else:
                pim = gdalObj.ReadAsArray(offsetCols, offsetRows, numCols, numRows)
            pim = np.moveaxis(pim, 0, -1)
            gdalObj = None
        except OSError as err:
            print(f'OS error: {err}')
        return pim

    @staticmethod
    def readPMask(pmaskPath, key):
        return PIMRead.readPMaskpatch(pmaskPath, key) 
    @staticmethod
    def readPMaskpatch(pmaskPath, key, offsetRows=0, offsetCols=0, numRows=-1, numCols=-1):
        try:
            with h5py.File(pmaskPath, 'r') as PmaskFile:
                if numRows == -1 and numCols == -1:
                    pmask = np.array(PmaskFile[key]).T
                else:
                    rows = slice(offsetRows, offsetRows+numRows)
                    cols = slice(offsetCols, offsetCols+numCols)
                    pmask = np.array(PmaskFile[key][cols,rows]).T # for some reason, you need to reverse rows and cols order.
            return pmask.astype(bool)
        except OSError as err:
            print(f'OS error: {err}')        
    
