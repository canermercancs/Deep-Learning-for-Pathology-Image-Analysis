"""
Pathology Image Class for Whole Slide Breast Histopathology Images.
Provides detailed functionality for pathology image objects.
e.g.
mycase  = 2352   # has to be and existing caseID.
pagenum = 7     # has to be within [3,9] range. The larger the number, the smaller the resolution of the slide 
pimObj  = PIM(mycase, pagenum)

##################
Caner Mercan, 2018
"""


import os.path
import warnings
import h5py                             # to read mat v7.3 files.
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
# my class imports
import DataNames as DN
from Directory import ImageDir
from Data import Cases, Polygons, Labels

# load data
Polygons.loadPolygonsMat()              # load Polygon Data
Labels.loadLabelsMat()                  # load Labels Data
Cases.loadCasesMat()                    # load Cases Data


class PIM(ImageDir): 
    """
    (P)athology (IM)age class.
    ....
    ####
    ### the order in which the code written in the constructor is important. DO NOT change unless sure.###
    ####
    """
    def __init__(self, pimID, page):

        super().__init__(page)

        self.pimID   = None
        self.pimName = None
        self.__setPIMID(pimID)     # checks and sets self.pimID 
        self.__setPIMName(pimID)   # sets self.pimName
        self.pimSize = []

        self.RGB     = PImage()    # only loaded when readRGB is called
        self.HE      = PImage()    # only loaded when readHE is called
        self.FGmask  = PMask()     # only loaded when readFGMask is called
        self.NUCmask = PMask()     # TODO!!! (NUCLEI MASK)
        
        self.ConsensusROIs      = [] 	# Consensus ROIs.
        self.SoftROIs           = [] 	# ROIs of the three experts.
        self.ConsensusDiagnoses = None 	# Consensus Diagnoses of the experts.
        self.ExpertDiagnoses    = {} 	# three experts; three diagnoses; each has their own diagnoses (dict)

        ##################
        self.__setPaths()          # also raise exception if RGB and HE paths do not exist.
        self.__setSize()           # sets pimSize 
        
        self.__setConsensusROIs()       # PIM consensus ROIs and diagnoses
        self.__setSoftROIs()            # PIM (soft)ROIs by each pathologist
        self.__setConsensusDiagnoses()  # PIM labels by each pathologist
        self.__setExpertDiagnoses()     # PIM labels by each pathologist

    def __setPIMID(self, pimID):
        if pimID not in Cases.IDS:
            raise Exception(f'CASE {pimID} DOES NOT EXIST!')
        self.pimID = pimID
    def __setPIMName(self, pimID):
        self.pimName = Cases.PAIRS[pimID]
    def __setPaths(self):
        self.RGB.setPath(self.dirRGB + self.pimName + DN.RGB_EXT) 
        self.HE.setPath(self.dirHE + self.pimName + DN.HE_EXT) 
        self.FGmask.setPath(self.dirFG + self.pimName + DN.FG_EXT) 
        if not (self.RGB.exist or self.HE.exist):
            raise Exception(f'CASE {self.pimID} RGB AND HE IMAGES MISSING!')
    def __setSize(self):
        path    = self.RGB.path if self.RGB.exist else self.HE.path
        Image   = gdal.Open(path)
        self.pimSize = [Image.RasterYSize, Image.RasterXSize]
    
    def __setExpertDiagnoses(self):
        caseIdx = Cases.IDS.index(self.pimID)
        for e, EXPERT_ID in enumerate(DN.EXPERT_ABBREV):
            expert_diags = {}
            for num_cls in Labels.NUM_CLASSES:
                expert_diags[num_cls] = Labels.expert_diags[num_cls][caseIdx][e]
            self.ExpertDiagnoses[EXPERT_ID] = ExpertDiagnoses(expert_diags, EXPERT_ID)
    def __setConsensusDiagnoses(self):
        caseIdx = Cases.IDS.index(self.pimID)
        consensus_diags = {}
        for num_cls in Labels.NUM_CLASSES:
            consensus_diags[num_cls] = Labels.consensus_diags[num_cls][caseIdx]
        self.ConsensusDiagnoses = ConsensusDiagnoses(consensus_diags)    
    def __setConsensusROIs(self):
        caseIdx     = Cases.IDS.index(self.pimID)
        cons_coords = Polygons.consensus_coords[caseIdx]
        for cons_coord in cons_coords:
            self.ConsensusROIs.append(ConsensusROI(cons_coord, self.page))      
    def __setSoftROIs(self):
        casepolyfilt = Polygons.soft_rects[:,0] == self.pimID
        polygons     = Polygons.polygons[casepolyfilt]
        softrects    = Polygons.soft_rects[casepolyfilt]
        for p in range(len(polygons)):
            self.SoftROIs.append(SoftROI(softrects[p,1], softrects[p,2], polygons[p], self.page))

    ### Reading .tif and .mat from file
    def readRGB(self):
        self.RGB.read()
    def readHE(self):
        self.HE.read()
    def readFGmask(self):
        self.FGmask.read()   
    ### Displaying .tif images and .mat masks
    def showRGB(self):
        self.RGB.show()
    def showHE(self):
        self.HE.show()
    def showFGmask(self):
        self.FGmask.show()

    ### Drawing/Displaying Expert ROIs
    def drawSoftROIs(self, expertID=DN.EXPERT_ABBREV):
        expertID = [expertID] if type(expertID)=='int' else expertID
        for ROI in self.SoftROIs:
            if ROI.expertID in expertID:
                ROI.draw()
    ### Drawing/Displaying Consensus ROIs
    def drawConsensusROIs(self):
        for ROI in self.ConsensusROIs:        
            ROI.draw()
    ### Printing/Displaying Expert Diagnoses
    def printExpertDiagnoses(self, num_classes=4):
        for expID, expDiag in self.ExpertDiagnoses.items():
            print(expDiag.print(num_classes))
    ### Printing/Displaying Consensus Diagnoses
    def printConsensusDiagnoses(self, num_classes=4):
        print(self.ConsensusDiagnoses.print(num_classes))


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
    #def readROI(self, roi):   
        #self.readpath(
        #return roi.read(self.path)
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


################################
### SoftROI and ConsensusROI ###
################################

class ROI():
    def __init__(self, page):
        self.page = page    
    def _poly2coord(self):
        x,y = np.min(self.polygon,0)
        xW, yW = np.max(self.polygon,0) - [x, y]
        x,y,xW,yW = int(x), int(y), int(xW), int(yW)
        return x,y,xW,xW
    def _coord2poly(self):
        x,y,xW,yW = self.coords[0], self.coords[1], self.coords[2], self.coords[3]
        polygon = np.array([[x,y],[x+xW,y],[x+xW,y+yW],[x,y+yW]])
        return polygon
    def draw(self, polygon, clrCode='k', width=2):
        polygon = np.vstack((polygon, polygon[0])) # close the polygon loop
        ys, xs  = zip(*polygon)
        plt.plot(xs,ys,clrCode, linewidth=width, alpha=.6)  
    def readPMask(self, maskPath):
        x,y,xW,yW = self._poly2coord()
        return PIMRead.readPMaskpatch(maskPath, DN.FGMASK_KEY, x,y,xW,yW)
    def readPImage(self, pimPath):
        x,y,xW,yW = self._poly2coord()
        return PIMRead.readPImagepatch(pimPath, x,y,xW,yW)
    def read(self, pimPath):
        isMat = pimPath[-4:] == '.mat'
        return self.readPMask(pimPath) if isMat else self.readPImage(pimPath)


class SoftROI(ROI):
    colorCode = ['r','b','g'] # grouped by actionID. zoom-in->red, slow_pannings->blue, fixation->green
    def __init__(self, expertID, actionID, polygon, page=8):
        super().__init__(page)
        self.expertID   = expertID
        self.actionID   = actionID
        self.polygon    = polygon
        self.coords     = None
        self._inpoints   = []    # points inside polygon. 
        self._outpoints  = []    # points outside polygon but inside surrounding rectangle.
        self.__setPolygon()
        self.__setCoords()
    def __setPolygon(self): # resize polygon based on page.
        self.polygon = self.polygon * (2**(3-self.page))
    def __setCoords(self):
        self.coords = self._poly2coord()
    def readFrom(self, pimPath):
        return super().read(pimPath)
    def draw(self):
        clrCode = SoftROI.colorCode[self.actionID-1] # actionID starts from 1.
        super().draw(self.polygon, clrCode, 4)
    #def set_inpoints(self):    
    #    self.polygon
    #def set_outpoints(self):

class ConsensusROI(ROI):
    def __init__(self, coords, page=8):
        super().__init__(page)
        self.coords     = None
        self.polygon    = None
        self.polygon_combined = None
        self.__setCoords(coords)
        self.__setPolygon()
    def __setCoords(self, coords):
        self.coords = np.array([coords[2], coords[1], coords[4], coords[3]])
        self.coords = self.coords * (2**(3-self.page))
    def __setPolygon(self):
        self.polygon = self._coord2poly()
    def readFrom(self, pimPath):
        return super().read(pimPath)
    def draw(self):
        super().draw(self.polygon, 'k--', 5)
    def __combinePolygons(self):
        """
        @TODO:
        will combine polygons into only one 'polygon_combined' polygon
        """
        #self.polygon_combined = ...
        pass
    

######################################
### Expert and Consensus Diagnoses ###
######################################
class Diagnoses():
    def __init__(self):
        pass
    def _getDiagnosticLabels(self, diags, classes):
        ind = np.where(diags)[0]
        return [classes[i] for i in ind]

class ConsensusDiagnoses(Diagnoses):
    def __init__(self, diagnoses):
        super().__init__()
        self.diagnoses  = diagnoses 
    def __addDiagnosis(self, num_cls, diagnosis):
        self.diagnoses[num_cls] = diagnosis
    def __repr__(self):
        return self.print()
    def print(self, num_cls=4):
        diagLabels = self._getDiagnosticLabels(self.diagnoses[num_cls], Labels.classes[num_cls])
        return 'Consensus Diagnoses: ' + ', '.join([d for d in diagLabels])    

class ExpertDiagnoses(Diagnoses):
    def __init__(self, diagnoses, expertID):
        super().__init__()
        self.diagnoses = diagnoses
        self.expertID = expertID
    def __repr__(self):
        return self.print()
    def print(self, num_cls=4):
        diagLabels = self._getDiagnosticLabels(self.diagnoses[num_cls], Labels.classes[num_cls])
        return f'Expert{self.expertID} Diagnoses: ' + ', '.join([d for d in diagLabels])



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
            if numRows == -1 or numCols == -1: # read the whole image
                pim = gdalObj.ReadAsArray()                        
            else:
                pim = gdalObj.ReadAsArray(offsetCols, offsetRows, numCols, numRows)
            pim = np.moveaxis(pim, 0, -1)
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
            return pmask
        except OSError as err:
            print(f'OS error: {err}')        
    

