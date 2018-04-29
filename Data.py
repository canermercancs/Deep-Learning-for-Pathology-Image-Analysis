"""
Data Loader for Pathology image data files.

##################
Caner Mercan, 2018.

"""


from Directory import DataDir as DD
import DataNames as DN
import scipy.io as sio


class Cases():
    """
    Loading all case IDs and names into memory.

    """
    IDS     = None
    NAMES   = None
    CV      = None
    PAIRS   = None

    @staticmethod
    def loadCasesMat():
        cases_struct = sio.loadmat(DD.DIR_DATA_SUPLEMENTARY + DN.CASES_MFILE)
        Cases.CV     = cases_struct[DN.CASECV_KEY]
        Cases.IDS    = cases_struct[DN.CASEIDS_KEY].flatten().tolist()
        Cases.NAMES  = [c[0][:-4] for c in cases_struct[DN.CASENAMES_KEY].flatten()]
        Cases.PAIRS  = {cID:cName for cID,cName in zip(Cases.IDS, Cases.NAMES)}   


class Polygons():
    """
    Loading all polygons (and soft_rects) into memory.
    """
    polygons    = None
    essentials  = None
    soft_rects  = None
    consensus_coords = None
    #unique_expertIDs = [] 
    #unique_actionIDs = []

    @staticmethod
    def loadPolygonsMat():
        Polygons.polygons       = Polygons.__loadFromPolygonMat(DN.POLYGONS_KEY)
        Polygons.essentials     = Polygons.__loadFromPolygonMat(DN.ESSENTIALS_KEY)
        Polygons.soft_rects     = Polygons.__loadFromPolygonMat(DN.SOFTRECT_KEY, flatten=False)
        Polygons.consensus_coords = Polygons.__loadFromDataMat(DN.CONSENSUS_COORDS_KEY)

        #Polygons.unique_expertIDs = np.unique(Polygons.soft_rects[:,1]).tolist()
        #Polygons.unique_actionIDs = np.unique(Polygons.soft_rects[:,2]).tolist()
    @staticmethod
    def __loadFromPolygonMat(KEY, flatten=True):
        """
        loads EXPERTs ROI polygons/soft_rects and etc. into memory
        """
        struct = sio.loadmat(DD.DIR_DATA + DN.POLYGONS_MFILE, variable_names=KEY)
        struct = struct[KEY].flatten() if flatten else struct[KEY]
        return struct
    @staticmethod
    def __loadFromDataMat(KEY):
        """
        loads Consensus ROI polygons into memory
        """
        struct = sio.loadmat(DD.DIR_DATA + DN.LABELS_MFILE, variable_names=KEY)
        struct = struct[KEY].flatten()
        return struct


class Labels():
    """
    Loads all class labels into memory.
    """
    NUM_CLASSES        = DN.NUM_CLASSES
    classes            = {}
    expert_diags       = {}
    consensus_diags    = {}

    def loadLabelsMat():
        for i,cls in enumerate(Labels.NUM_CLASSES):
            Labels.classes[cls]          = Labels.__loadFromDataMat(DN.CLASSES_KEYS[i] , cell=1)
            Labels.expert_diags[cls]     = Labels.__loadFromDataMat(DN.EXPERT_DIAGS_KEYS[i], cell=2)
            Labels.consensus_diags[cls]  = Labels.__loadFromDataMat(DN.CONSENSUS_DIAGS_KEYS[i], cell=1)

    def __loadFromDataMat(KEY, cell=0):
        """
        cell denotes the number of levels inside the struct.
        """
        struct = sio.loadmat(DD.DIR_DATA + DN.LABELS_MFILE, variable_names=KEY)
        struct = struct[KEY]
        if cell==2: 
            # EXPERT structs contain 3 expert opinions; require different type of 'flatten'
            struct = [[val[i].flatten() for i in range(len(val)) ] for val in struct]
        elif cell==1:        
            struct = [val[0] for val in struct.flatten()]
        else:
            struct = struct.flatten().tolist()         
        return struct









    




