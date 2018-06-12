"""
Pathology Image Class for Whole Slide Breast Histopathology Images.
Provides detailed functionality for pathology image objects.
e.g.
mycase  = 2352   # has to be and existing caseID.
pagenum = 7     # has to be within [3,9] range. The larger the number, the smaller the resolution of the slide 
pimObj  = PIM(mycase, pagenum)

##################
License: Apache License 2.0
author: Caner Mercan, 2018
"""

import os
import warnings
import numpy as np
import gdal
# my class imports
from . import dataNames as DN
from .directory import ImageDir
from .data import Cases, Polygons, Labels
from .roi import ConsensusROI, SoftROI
from .pim_meta import PImage, PMask

class PIM(ImageDir): 
    """
    (P)athology (IM)age class.
    ....
    ####
    ### the order in which the code written in the constructor is important. DO NOT change unless sure.###
    ####
    """
    def __init__(self, pimID, page=8):

        super().__init__(page)
        self.__load_data__()
        
        self.pimID   = None
        self.__setPIMID__(pimID)     # checks and sets self.pimID 
        self.pimName = None
        self.__setPIMName__()       # sets self.pimName
        self.pimSize = []
        self.pimCV   = -1          # which fold this case belongs to.

        self.RGB     = PImage()    # only loaded when readRGB is called
        self.HE      = PImage()    # only loaded when readHE is called
        self.FGmask  = PMask()     # only loaded when readFGMask is called
        self.NUCmask = PMask()     # TODO!!! (NUCLEI MASK)
        
        self.ConsensusROIs      = [] 	# Consensus ROIs.
        self.SoftROIs           = [] 	# ROIs of the three experts.
        self.ConsensusDiagnoses = None 	# Consensus Diagnoses of the experts.
        self.ExpertDiagnoses    = {} 	# three experts; three diagnoses; each has their own diagnoses (dict)

        ##################
        self.__setPaths__()          # also raise exception if RGB and HE paths do not exist.
        self.__setSize__()           # sets pimSize 
        self.__setCV__()
        
        self.__setConsensusROIs()       # PIM consensus ROIs and diagnoses
        self.__setSoftROIs()            # PIM (soft)ROIs by each pathologist
        self.__setConsensusDiagnoses()  # PIM labels by each pathologist
        self.__setExpertDiagnoses()     # PIM labels by each pathologist

    def __load_data__(self):        
        # load data
        Polygons.loadPolygonsMat()              # load Polygon Data
        Labels.loadLabelsMat()                  # load Labels Data
        Cases.loadCasesMat()                    # load Cases Data
    def __setPIMID__(self, pimID):
        if pimID not in Cases.IDS:
            raise Exception(f'CASE {pimID} DOES NOT EXIST!')
        self.pimID = pimID
    def __setPIMName__(self):
        self.pimName = Cases.PAIRS[self.pimID]
    def __setPaths__(self):
        self.RGB.setPath(os.path.join(self.dirRGB, self.pimName + DN.RGB_EXT))
        self.HE.setPath(os.path.join(self.dirHE, self.pimName + DN.HE_EXT))
        self.FGmask.setPath(os.path.join(self.dirFG, self.pimName + DN.FG_EXT)) 
        if not (self.RGB.exist or self.HE.exist):
            raise Exception(f'CASE {self.pimID} RGB AND HE IMAGES MISSING!')
    def __setSize__(self):
        path         = self.RGB.path if self.RGB.exist else self.HE.path
        gdalObj      = gdal.Open(path)
        self.pimSize = [gdalObj.RasterYSize, gdalObj.RasterXSize]
        gdalObj      = None
    def __setCV__(self):
        self.pimCV = Cases.CV[Cases.IDS.index(self.pimID)].index(1) 
    
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
        caseIdx = Cases.IDS.index(self.pimID)
        coords  = Polygons.consensus_coords[caseIdx]
        paths   = [self.RGB.path, self.HE.path, self.FGmask.path]
        for coord in coords:
            self.ConsensusROIs.append(ConsensusROI(coord, paths, self.page))      
    def __setSoftROIs(self):
        casepolyfilt = Polygons.soft_rects[:,0] == self.pimID
        polygons     = Polygons.polygons[casepolyfilt]
        essentials   = Polygons.essentials[casepolyfilt]
        softrects    = Polygons.soft_rects[casepolyfilt]
        paths        = [self.RGB.path, self.HE.path, self.FGmask.path]
        for p in range(len(polygons)):
            self.SoftROIs.append(SoftROI(softrects[p,1], softrects[p,2], polygons[p], essentials[p], paths, self.page))

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
    def drawSoftROIs(self, expertID=DN.EXPERT_ABBREV, onlyEssentials=False, width=3, color=None):
        expertID = [expertID] if isinstance(expertID, int) else expertID
        for ROI in self.SoftROIs:
            if ROI.expertID in expertID: 
                if ROI.isEssential or not onlyEssentials:
                    ROI.draw(width=width, color=color)
    ### Drawing/Displaying Consensus ROIs
    def drawConsensusROIs(self, width=3, color=None):
        for ROI in self.ConsensusROIs:        
            ROI.draw(width=width, color=color)
    ### Printing/Displaying Expert Diagnoses
    def printExpertDiagnoses(self, num_classes=4):
        for expID, expDiag in self.ExpertDiagnoses.items():
            print(expDiag.toString(num_classes))
    ### Printing/Displaying Consensus Diagnoses
    def printConsensusDiagnoses(self, num_classes=4):
        print(self.ConsensusDiagnoses.toString(num_classes))


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
        return self.toString()
    def toString(self, num_cls=4):
        diagLabels = self._getDiagnosticLabels(self.diagnoses[num_cls], Labels.classes[num_cls])
        return 'Consensus Diagnoses: ' + ', '.join([d for d in diagLabels])    

class ExpertDiagnoses(Diagnoses):
    def __init__(self, diagnoses, expertID):
        super().__init__()
        self.diagnoses = diagnoses
        self.expertID = expertID
    def __repr__(self):
        return self.toString()
    def toString(self, num_cls=4):
        diagLabels = self._getDiagnosticLabels(self.diagnoses[num_cls], Labels.classes[num_cls])
        return f'Expert{self.expertID} Diagnoses: ' + ', '.join([d for d in diagLabels])


