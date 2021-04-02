import numpy as np
from numpy.testing import assert_almost_equal

from psnr_hvsm.bt601 import bt601luma, bt601luma_norm


def test_bt601luma():
    x = np.array([92, 4, 110], dtype=float).reshape(1, 1, 3)

    y = bt601luma(x).item()

    assert(y*255 == 55)  # value generated using rgb2ycbcr in octave


def test_bt601luma_norm():
    x = np.array([92, 4, 110], dtype=float).reshape(1, 1, 3) / 255

    y = bt601luma_norm(x).item()

    assert_almost_equal(y, 1.793176470588235e-01)  # value generated using rgb2ycbcr in octave, then renormalised to [0,1]
