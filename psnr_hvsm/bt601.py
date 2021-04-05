"""Functions for converting RGB to normalized BT.601 YCbCr and YPbPr."""

import numpy as np


def bt601ycbcr(a):
    """Convert an RGB image into normalized YCbCr."""
    r, g, b = np.moveaxis(a, 2, 0).astype(np.float64)
    y = np.round(16 + 65.481*r/255 + 128.553*g/255 + 24.966*b/255)/255
    cb = np.round(128 - 37.797*r/255 - 74.203*g/255 + 112.0*b/255)/255
    cr = np.round(128 + 112.0*r/255 - 93.786*g/255 - 18.214*b/255)/255
    return y, cb, cr


def bt601ypbpr(a):
    """Convert a normalized RGB image into YPbPr (analog YCbCr)."""
    r, g, b = np.moveaxis(a, 2, 0).astype(np.float64)
    kb = 0.114
    kr = 0.299
    y = kr*r + (1-kr-kb)*g + kb*b
    pb = -(kr/(2-2*kb))*r - (1-kr-kb)/(2-2*kb)*g + 0.5*b + 0.5
    pr = 0.5*r - (1-kr-kb)/(2-2*kr)*g - (kb/(2-2*kr)) * b + 0.5
    return y, pb, pr
