"""
Pathology Image ROI Class for Whole Slide Breast Histopathology Images.
Provides detailed functionality for pathology image roi objects.

##################
License: Apache License 2.0
author: Caner Mercan, 2018
"""

import matplotlib.pyplot as plt
from . import dataNames as DN
from .pim_meta import PImage, PMask
from .utils.pim_reader import PIMRead
from .utils.pim_patch_sampler import *

################################
### SoftROI and ConsensusROI ###
################################

class ROI():
    def __init__(self, paths, page):
        self.page   = page    
        self.RGB    = PImage(None, paths[0])    
        self.HE     = PImage(None, paths[1])    
        self.FGmask = PMask(None, paths[2])     
    def _resize(self, mat2resize):
        return mat2resize * (2.0**(3-self.page))        
    def _poly2coord(self):
        x,y = np.min(self.polygon,0)
        xW, yW = np.max(self.polygon,0) - [x, y]
        x,y,xW,yW = int(x), int(y), int(xW), int(yW)
        return x,y,xW,yW
    def _coord2poly(self):
        x,y,xW,yW = self.coords[0], self.coords[1], self.coords[2], self.coords[3]
        polygon = np.array([[x,y],[x+xW,y],[x+xW,y+yW],[x,y+yW]])
        return polygon
    def _draw(self, polygon, clrCode='k', width=2):
        polygon = np.vstack((polygon, polygon[0])) # close the polygon loop
        ys, xs  = zip(*polygon)
        plt.plot(xs,ys,clrCode, linewidth=width, alpha=.6)  
    def _polyArea_(self):
        x,y = self.polygon[:,0], self.polygon[:,1]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    def _readPMask(self, maskPath):
        x,y,xW,yW = self._poly2coord()
        # currently, background masks available only for pages 3 and 7.
        if self.page in [3,7]: 
            return PIMRead.readPMaskpatch(maskPath, DN.FGMASK_KEY, x,y,xW,yW)
        else:
            return np.ones((xW, yW)).astype(bool)
    def _readPImage(self, pimPath):
        x,y,xW,yW = self._poly2coord()
        return PIMRead.readPImagepatch(pimPath, x,y,xW,yW)
    def _read(self, pimPath):
        isMat = pimPath[-4:] == '.mat'
        return self._readPMask(pimPath) if isMat else self._readPImage(pimPath)
    
    def _cropPoiPatch(self, pimPath, win_size, num_patches):
        img         = self._read(pimPath)    
        img_he      = self._read(self.HE.path)
        img_mask    = self._read(self.FGmask.path)  
        offset      = tuple(w//2 for w in win_size)
        roi_mask    = getNucleiROI(img_he[:,:,0], self.page)
        poi_idx     = getPOIs(roi_mask, offset)
        patch_iter  = cropPatchfromImage(num_patches, poi_idx, win_size, img, img_mask) #yields patches one by one.
        return patch_iter      

class SoftROI(ROI):
    colorCode = ['r','b','g'] # grouped by actionID. zoom-in->red, slow_pannings->blue, fixation->green
    def __init__(self, expertID, actionID, polygon, polygon_filters, paths, page=8):
        super().__init__(paths, page)
        self.expertID    = expertID
        self.actionID    = actionID
        self.polygon     = polygon
        self.filters     = polygon_filters
        self.coords      = None
        # self._inpoints   = []    # points inside polygon. 
        # self._outpoints  = []    # points outside polygon but inside surrounding rectangle.
        self.__setPolygon()
        self.__setCoords()

    def __setPolygon(self): # resize polygon based on page.
        self.polygon[self.polygon < 0] = 0 # if there are any negative values, correct them.
        self.polygon = self._resize(self.polygon)
    def __setCoords(self):
        self.coords = self._poly2coord()
    def readFrom(self, pimPath):
        return super()._read(pimPath)
    def draw(self, width=3, color=None):
        color = SoftROI.colorCode[self.actionID-1] if not color else color  # actionID starts from 1.
        super()._draw(self.polygon, color, width)
    def crop(self, pimPath, win_size, num_patches):
        return super()._cropPoiPatch(pimPath, win_size, num_patches)
    def polyArea(self):
        return super()._polyArea_()

    #def set_inpoints(self):    
    #    self.polygon
    #def set_outpoints(self):

class ConsensusROI(ROI):
    def __init__(self, coords, paths, page=8):
        super().__init__(paths, page)
        self.coords     = None
        self.polygon    = None
        self.polygon_combined = None
        self.__setCoords(coords)
        self.__setPolygon()
    def __setCoords(self, coords):
        self.coords = np.array([coords[2], coords[1], coords[4], coords[3]])
        self.coords[self.coords < 0] = 0  # if there are any negative values, correct them.
        self.coords = self._resize(self.coords)
    def __setPolygon(self):
        self.polygon = self._coord2poly()
    def readFrom(self, pimPath):
        return super()._read(pimPath)
    def draw(self, width=3, color=None):
        color = 'k--' if not color else color  # actionID starts from 1.
        super()._draw(self.polygon, color, width)
    def crop(self, pimPath, win_size, num_patches):
        return super()._cropPoiPatch(pimPath, win_size, num_patches)
    def polyArea(self):
        return super()._polyArea_()
    def __combinePolygons(self):
        """
        @TODO:
        will combine polygons into only one 'polygon_combined' polygon
        """
        #self.polygon_combined = ...
        pass
    
