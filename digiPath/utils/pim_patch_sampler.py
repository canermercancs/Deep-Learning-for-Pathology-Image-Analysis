"""
Pathology Image Patch Sampling for Whole Slide Breast Histopathology Images/Masks.
Provides functionality for pathology image patch sampling functions; 
nuclei dense region detection, and cropping/sampling patches based on points of interest.

##################
License: Apache License 2.0
author: Caner Mercan, 2018
"""

import pdb
import numpy as np
import scipy.ndimage as ndimage

##################################################################
### pathology image sampling based from nucleus dense regions ###
##################################################################

def getNucleiROI(img_gray, page_size):
    """
    returns nuclei regions using low pass Gaussian filter.
    img_gray should be the first channel of HE image for nuclei POI detection.
    e.g.
        pimObj      = PIM(2352, 5)
        consy       = pimObj.ConsensusROIs[0]
        consy_HE    = consy.readFrom(pimObj.HE.path)
        nuclei_roi  = getNucleiROI(consy_HE[:,:,0], pimObj.page)
    """
    assert len(img_gray.shape) == 2
    assert page_size in range(3,10)

    # sigma=20 when page_size==3, sigma=1.25 when page_size==7, etc.
    sigma       = (1/2**page_size) * 160 
    img_gauss   = (ndimage.gaussian_filter(img_gray, sigma) < 200)
    return img_gauss

def getPOIs(roi_mask, offset_size):
    """
    get POIs from an ROI mask
    e.g.
        pimObj      = PIM(2352, 5)
        consy       = pimObj.ConsensusROIs[0]
        consy_HE    = consy.readFrom(pimObj.HE.path)
        nuclei_roi  = getNucleiROI(consy_HE[:,:,0], pimObj.page)
        nuclei_poi  = getPOIs(nuclei_roi, offset_size)
    """    
    assert isinstance(offset_size, (int, tuple))
    if isinstance(offset_size, int):
        offset_size = (offset_size, offset_size)
    else:
        assert len(offset_size) == 2
    
    off_h, off_w    = [off for off in offset_size]
    h, w            = roi_mask.shape[:2]
    h_go, h_end     = off_h, h - off_h
    w_go, w_end     = off_w, w - off_w

    # all point of interests
    poi = np.array(np.where(roi_mask[h_go:h_end, w_go:w_end]==True))
    poi = np.array([poi[0]+off_h, poi[1]+off_w])
    return poi


def cropPatchfromImage(num_samples, poi, patch_size, image, image_mask=None):
    """
    crop random image patches using poi information.
    poi is x,y coordinate(s) which are the centers of the output patches.
    e.g.
        pimObj      = PIM(2352, 5)
        consy       = pimObj.ConsensusROIs[0]
        consy_HE    = consy.readFrom(pimObj.HE.path)
        consy_mask  = consy.readFrom(pimObj.FGmask.path)

        nuclei_roi  = getNucleiROI(consy_HE[:,:,0], pimObj.page)
        nuclei_poi  = getPOIs(nuclei_roi, offset_size)
        patch_iter  = cropPatchfromImage(200, nuclei_poi, (224,224), consy_HE, consy_mask)

        patch, coor = next(patch_iter).values()
        plt.imshow(patch)
        plt.show()
        print(coor)
    """    
    num_points = len(poi[0])
    if num_points == 0:
        yield None

    assert num_samples > 0 and num_points > 0
    assert len(image.shape) >= 2
    assert isinstance(patch_size, (int, tuple))
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    else:
        assert len(patch_size) == 2
    if num_points < num_samples:
        print(f'# POIs({num_points}) are less than requested sample size({num_samples})')
        num_samples = num_points

    off_row, off_col = [off // 2 for off in patch_size]
    idx = np.random.randint(0, num_points, num_samples)
#    # more correct way but takes more time.
    # idx = np.random.permutation(range(0, len(poi[0])))[:num_samples]    
    for i in idx:
        row_go, row_end = poi[0,i]-off_row, poi[0,i]+off_row
        col_go, col_end = poi[1,i]-off_col, poi[1,i]+off_col
        patch_img = image[row_go:row_end, col_go:col_end, :]
        if image_mask is not None:
            mask  = image_mask[row_go:row_end, col_go:col_end]
            patch_img[~mask] = 0
        patch = {
            'img': patch_img,
            'loc': [row_go, col_go, row_end-row_go, col_end-col_go]} #x,y,xW,yW 
        yield patch
