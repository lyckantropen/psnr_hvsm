import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.fft import dctn as sp_dctn

from psnr_hvsm import psnr_hvsm, hvsm_mse

import cv2

org = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE).astype(float) #/ 255
tst = cv2.imread('baboon_msk.png', cv2.IMREAD_GRAYSCALE).astype(float) #/ 255

def _get_psnr(s, mv):
    if s == 0:
        return 100
    else:
        return min(10.0 * np.log10(mv**2/s), 100.0)

mse = hvsm_mse(org, tst)
v = psnr_hvsm(org, tst)

print(v)

