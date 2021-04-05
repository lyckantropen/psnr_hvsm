import numpy as np
from numpy.testing import assert_almost_equal

from psnr_hvsm.bt601 import bt601ycbcr, bt601ypbpr


def test_bt601ycbcr():
    x = np.array([110, 4, 92], dtype=float).reshape(1, 1, 3)

    y, cb, cr = bt601ycbcr(x)

    assert(y.item() == 55/255)  # generated in octave
    assert(cb.item() == 151/255)  # generated in octave
    assert(cr.item() == 168/255)  # generated in octave


def test_bt601ypbpr():
    x = np.array([110, 4, 92], dtype=float).reshape(1, 1, 3) / 255

    y, pb, pr = bt601ypbpr(x)

    assert_almost_equal(y.item(), 1.793176470588235e-01)  # value generated using rgb2ycbcr in octave, then renormalised to [0,1]
    assert_almost_equal(pb.item(), 6.024078254326561e-01)  # value generated using rgb2ycbcr in octave, then renormalised to [0,1]
    assert_almost_equal(pr.item(), 6.797823837095466e-01)  # value generated using rgb2ycbcr in octave, then renormalised to [0,1]
