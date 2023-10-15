import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from psnr_hvsm.numpy import bt601ycbcr as bt601ycbcr_np
from psnr_hvsm.numpy import bt601ypbpr as bt601ypbpr_np
from psnr_hvsm.torch import bt601ycbcr as bt601ycbcr_pt
from psnr_hvsm.torch import bt601ypbpr as bt601ypbpr_pt


@pytest.mark.parametrize('bt601ycbcr_impl', [bt601ycbcr_np, bt601ycbcr_pt], ids=['numpy', 'torch'])
def test_bt601ycbcr(bt601ycbcr_impl):
    x = np.array([110, 4, 92], dtype=float).reshape(1, 1, 3)

    y, cb, cr = bt601ycbcr_impl(x)

    assert y.item() == 55 / 255   # generated in octave
    assert cb.item() == 151 / 255   # generated in octave
    assert cr.item() == 168 / 255   # generated in octave


@pytest.mark.parametrize('bt601ypbpr_impl', [bt601ypbpr_np, bt601ypbpr_pt], ids=['numpy', 'torch'])
def test_bt601ypbpr(bt601ypbpr_impl):
    x = np.array([110, 4, 92], dtype=float).reshape(1, 1, 3) / 255

    y, pb, pr = bt601ypbpr_impl(x)

    assert_almost_equal(y.item(), 1.793176470588235e-01)  # value generated using rgb2ycbcr in octave, then renormalised to [0,1]
    assert_almost_equal(pb.item(), 6.024078254326561e-01)  # value generated using rgb2ycbcr in octave, then renormalised to [0,1]
    assert_almost_equal(pr.item(), 6.797823837095466e-01)  # value generated using rgb2ycbcr in octave, then renormalised to [0,1]
