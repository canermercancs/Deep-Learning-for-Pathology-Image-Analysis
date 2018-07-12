"""
@author: Caner Mercan
"""

import random
import numpy as np

class RandomRGBJitter(object):
    """
    """
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, jitt=20):
        if random.random() < self.p:
            noise = np.random.randint(0, jitt, img.shape)  # design jitter/noise here
            im1 = img.astype('uint16') # to cope with overflowing
            im2 = noise.astype('uint16') # to cope with overflowing
            tmp = im1 + im2
            img = tmp.clip(0, 255).astype('uint8')
        return img


class RandomHueJitter(self, p=0.5):
    """
    Randomly perturb Hue channel in HSV image
    """
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, rgb, jitt=0.1):
        if self.p < 0.5:
            # convert rgb to hsv
            hsv = rgb2hsv_fast(rgb)
            # generate random noise 
            noise = np.random.random((rgb.shape[0], rgb.shape[1])) * jitt
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
        return rgb

