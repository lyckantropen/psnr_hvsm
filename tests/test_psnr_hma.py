from pathlib import Path

import pytest
from imageio.v2 import imread
from numpy.testing import assert_almost_equal

from psnr_hvsm import bt601ycbcr, ha_hma_mse, psnr_ha_hma
from psnr_hvsm._psnr_hvsm import hvs_hvsm_mse as hvs_hvsm_mse_cpp
from psnr_hvsm.numpy import ha_hma_mse as ha_hma_mse_np
from psnr_hvsm.numpy import hvs_hvsm_mse as hvs_hvsm_mse_np
from psnr_hvsm.numpy import psnr_ha_hma as psnr_ha_hma_np
try:
    from psnr_hvsm.torch import ha_hma_mse as ha_hma_mse_pt
    from psnr_hvsm.torch import hvs_hvsm_mse as hvs_hvsm_mse_pt
    from psnr_hvsm.torch import psnr_ha_hma as psnr_ha_hma_pt
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    ha_hma_mse_pt = None
    hvs_hvsm_mse_pt = None
    psnr_ha_hma_pt = None

test_dir = Path(__file__).parent.absolute()


@pytest.mark.parametrize('hvs_hvsm_mse_impl', [
    pytest.param(hvs_hvsm_mse_cpp, id='cpp'),
    pytest.param(hvs_hvsm_mse_np, id='numpy'),
    pytest.param(hvs_hvsm_mse_pt, id='torch', marks=pytest.mark.skipif(hvs_hvsm_mse_pt is None, reason='PyTorch is not installed')),
])
def test_psnr_hma(hvs_hvsm_mse_impl):
    org = imread((test_dir / 'kodim01.png').as_posix()).astype(float) / 255
    tst = imread((test_dir / 'kodim01_contrast.png').as_posix()).astype(float) / 255

    psnr_ha, psnr_hma = psnr_ha_hma_np(org, tst, hvs_hvsm_mse_impl=hvs_hvsm_mse_impl)

    assert_almost_equal(psnr_ha, 23.72516252033972)  # result obtained in octave
    assert_almost_equal(psnr_hma, 24.30979923387402, decimal=4)  # result obtained in octave


@pytest.mark.parametrize('ha_hma_mse_impl', [
    pytest.param(ha_hma_mse_np, id='numpy'),
    pytest.param(ha_hma_mse_pt, id='torch', marks=pytest.mark.skipif(ha_hma_mse_pt is None, reason='PyTorch is not installed')),
])
def test_ha_hma_mse(ha_hma_mse_impl):
    org = imread((test_dir / 'kodim01.png').as_posix()).astype(float)
    tst = imread((test_dir / 'kodim01_contrast.png').as_posix()).astype(float)

    mse_ha, mse_hma = ha_hma_mse_impl(org, tst)

    assert_almost_equal(mse_ha, 275.7808519088271)  # result obtained in octave
    assert_almost_equal(mse_hma, 241.0462821953413, decimal=3)  # result obtained in octave


@pytest.mark.parametrize('ha_hma_mse_impl', [
    pytest.param(ha_hma_mse_np, id='numpy'),
    pytest.param(ha_hma_mse_pt, id='torch', marks=pytest.mark.skipif(ha_hma_mse_pt is None, reason='PyTorch is not installed')),
])
def test_ha_hma_mse_color(ha_hma_mse_impl):
    org = imread((test_dir / 'kodim01_c.png').as_posix()).astype(float)
    tst = imread((test_dir / 'kodim01_c_contrast.png').as_posix()).astype(float)

    y0, _, _ = bt601ycbcr(org)
    y1, _, _ = bt601ycbcr(tst)

    mse_ha, mse_hma = ha_hma_mse_impl(y0 * 255, y1 * 255)

    assert_almost_equal(mse_ha, 314.1129805328927)  # result obtained in octave
    assert_almost_equal(mse_hma, 278.3264049014794, decimal=2)  # result obtained in octave
