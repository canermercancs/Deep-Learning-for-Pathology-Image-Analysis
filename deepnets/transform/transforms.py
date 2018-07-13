"""
@author: Caner Mercan
"""

import random
import numpy as np
from PIL import Image
from digiPath.utils.image_transformation import *

class RandomRGBJitter(object):
    """
    @TODO: write these w.r.t to PIL class. Too slow currently due to all the transformations.
    
    Randomly perturbs all of the image channels
    jitt: Perturbation amount
    """
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, jitt=20):
        if random.random() < self.p:
            img = np.asarray(img, dtype=np.uint8)
            noise = np.random.randint(0, jitt, img.shape)  # design jitter/noise here
            im1 = img.astype('uint16') # to cope with overflowing
            im2 = noise.astype('uint16') # to cope with overflowing
            tmp = im1 + im2
            img = tmp.clip(0, 255).astype('uint8')
            img = Image.fromarray(img, 'RGB')
        return img


class RandomHueJitter(object):
    """
    @TODO: write these w.r.t to PIL class. Too slow currently due to all the transformations.

    Randomly perturb Hue channel of a given RGB image
    jitt: Perturbation weigth
    i.e. new_hue = [-jitt*hue, jitt*hue]
    """
    def __init__(self, p=0.5, jitt=0.2):
        self.p = p
        self.jitt = jitt
    def __call__(self, rgb):
        if self.p < 0.5:
            # convert rgb to hsv
            rgb = np.asarray(rgb, dtype=np.uint8)
            hsv = rgb2hsv_fast(rgb)
            # generate random noise 
            noise = np.random.random((rgb.shape[0], rgb.shape[1])) * (np.random.rand()*2*self.jitt - self.jitt)
            jitter = np.zeros_like(hsv)
            # align the noise to the hue channel (1st channel)
            jitter[:,:,0] = noise
            # combine the hsv and the 
            im = (hsv + jitter)
            # clip hue to make it between 0 and 1
            hue = np.clip(im[:,:,0], 0, 1)
            # combine hue and the rest together
            im = np.stack([hue, im[:,:,1], im[:,:,2]], axis=2)
            # convert the hsv to back to rgb           
            rgb = hsv2rgb_fast(im) 
            rgb = Image.fromarray(rgb, 'RGB')
        return rgb

