import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from psnr_hvsm.numpy import bt601ycbcr as bt601ycbcr_np
from psnr_hvsm.numpy import bt601ypbpr as bt601ypbpr_np
try:
    from psnr_hvsm.torch import bt601ycbcr as bt601ycbcr_pt
    from psnr_hvsm.torch import bt601ypbpr as bt601ypbpr_pt
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    bt601ycbcr_pt = None
    bt601ypbpr_pt = None


@pytest.mark.parametrize('bt601ycbcr_impl', [
    pytest.param(bt601ycbcr_np, id='numpy'),
    pytest.param(bt601ycbcr_pt, id='torch', marks=pytest.mark.skipif(bt601ycbcr_pt is None, reason='PyTorch is not installed')),
])
def test_bt601ycbcr(bt601ycbcr_impl):
    x = np.array([110, 4, 92], dtype=float).reshape(1, 1, 3)

    y, cb, cr = bt601ycbcr_impl(x)

    assert y.item() == 55 / 255   # generated in octave
    assert cb.item() == 151 / 255   # generated in octave
    assert cr.item() == 168 / 255   # generated in octave


@pytest.mark.parametrize('bt601ypbpr_impl', [
    pytest.param(bt601ypbpr_np, id='numpy'),
    pytest.param(bt601ypbpr_pt, id='torch', marks=pytest.mark.skipif(bt601ypbpr_pt is None, reason='PyTorch is not installed')),
])
def test_bt601ypbpr(bt601ypbpr_impl):
    x = np.array([110, 4, 92], dtype=float).reshape(1, 1, 3) / 255

    y, pb, pr = bt601ypbpr_impl(x)

    assert_almost_equal(y.item(), 1.793176470588235e-01)  # value generated using rgb2ycbcr in octave, then renormalised to [0,1]
    assert_almost_equal(pb.item(), 6.024078254326561e-01)  # value generated using rgb2ycbcr in octave, then renormalised to [0,1]
    assert_almost_equal(pr.item(), 6.797823837095466e-01)  # value generated using rgb2ycbcr in octave, then renormalised to [0,1]


@pytest.mark.parametrize('bt601ycbcr_impl', [
    pytest.param(bt601ycbcr_np, id='numpy'),
    pytest.param(bt601ycbcr_pt, id='torch', marks=pytest.mark.skipif(bt601ycbcr_pt is None, reason='PyTorch is not installed')),
])
def test_bt601ycbcr_float_precision(bt601ycbcr_impl):
    x = np.array([110, 4, 92], dtype=float).reshape(1, 1, 3)

    y, cb, cr = bt601ycbcr_impl(x, precision='float')

    # expected values computed with exact rational arithmetic (fractions.Fraction),
    # independently of the float64 pipeline, for the same BT.601 matrix without rounding
    assert_almost_equal(y.item(), 2.1674731257208765e-01, decimal=12)
    assert_almost_equal(cb.item(), 5.919187697039601e-01, decimal=12)
    assert_almost_equal(cr.item(), 6.598872433679354e-01, decimal=12)


@pytest.mark.parametrize('bt601ycbcr_impl', [
    pytest.param(bt601ycbcr_np, id='numpy'),
    pytest.param(bt601ycbcr_pt, id='torch', marks=pytest.mark.skipif(bt601ycbcr_pt is None, reason='PyTorch is not installed')),
])
def test_bt601ycbcr_precision_divergence(bt601ycbcr_impl):
    # the rounding step changes luma and both chroma components for this input (issue #9)
    x = np.array([110, 4, 92], dtype=float).reshape(1, 1, 3)

    y8, cb8, cr8 = bt601ycbcr_impl(x)
    yf, cbf, crf = bt601ycbcr_impl(x, precision='float')

    assert y8.item() != yf.item()
    assert cb8.item() != cbf.item()
    assert cr8.item() != crf.item()


@pytest.mark.parametrize('bt601ycbcr_impl', [
    pytest.param(bt601ycbcr_np, id='numpy'),
    pytest.param(bt601ycbcr_pt, id='torch', marks=pytest.mark.skipif(bt601ycbcr_pt is None, reason='PyTorch is not installed')),
])
def test_bt601ycbcr_invalid_precision(bt601ycbcr_impl):
    x = np.array([110, 4, 92], dtype=float).reshape(1, 1, 3)

    with pytest.raises(ValueError):
        bt601ycbcr_impl(x, precision='16-bit')


@pytest.mark.skipif(bt601ycbcr_pt is None, reason='PyTorch is not installed')
@pytest.mark.parametrize('precision', ['8-bit', 'float'])
def test_bt601ycbcr_backends_agree(precision):
    # the numpy implementation accepts a single 3-D image only, so the input stays 3-D;
    # the input is float64 because torch promotes integer input to its default dtype
    # (float32), while the numpy implementation always computes in float64
    x = (np.arange(768).reshape(4, 64, 3) * 97 % 256).astype(np.float64)

    np_y, np_cb, np_cr = bt601ycbcr_np(x, precision=precision)
    pt_y, pt_cb, pt_cr = bt601ycbcr_pt(x, precision=precision)

    assert np.array_equal(pt_y.numpy(), np_y)
    assert np.array_equal(pt_cb.numpy(), np_cb)
    assert np.array_equal(pt_cr.numpy(), np_cr)


@pytest.mark.parametrize('bt601ypbpr_impl', [
    pytest.param(bt601ypbpr_np, id='numpy'),
    pytest.param(bt601ypbpr_pt, id='torch', marks=pytest.mark.skipif(bt601ypbpr_pt is None, reason='PyTorch is not installed')),
])
@pytest.mark.parametrize('precision', ['8-bit', 'float'])
def test_bt601ypbpr_precision_is_noop(bt601ypbpr_impl, precision):
    # YPbPr is never rounded, so the precision option is accepted but has no effect
    x = np.array([110, 4, 92], dtype=float).reshape(1, 1, 3) / 255

    default = bt601ypbpr_impl(x)
    with_precision = bt601ypbpr_impl(x, precision=precision)

    for component, default_component in zip(with_precision, default):
        assert np.array_equal(np.asarray(component), np.asarray(default_component))
