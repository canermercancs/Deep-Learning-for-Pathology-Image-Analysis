
import numpy as np
import colorsys
rgb_to_hsv=np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb=np.vectorize(colorsys.hsv_to_rgb)

def rgb2hsv(rgb):
    """
    convert rgb to hsv (slow but reliable built-in method for conversion)
    """
    r,g,b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    h,s,v = rgb_to_hsv(r,g,b)
    hsv = np.stack((h,s,v), axis=2)
    return hsv

def hsv2rgb(hsv):
    """
    convert hsv to rgb (slow but reliable built-in method for conversion)
    """
    h,s,v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    r,g,b = hsv_to_rgb(h,s,v)
    rgb = np.stack((r,g,b), axis=2)
    return rgb.astype('uint8')


def rgb2hsv_fast(rgb):
    """
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    source: https://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil
    """
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv2rgb_fast(hsv):
    """
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    source: https://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil
    """
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def shift_hue(arr,hout):
    """
    hsv = rgb_to_hsv(rgb)
    arr = np.array(roi_hsv)
    hue = (180)/360.0
    new_img = shift_hue(arr,hue)
    source: https://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil
    """
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb
