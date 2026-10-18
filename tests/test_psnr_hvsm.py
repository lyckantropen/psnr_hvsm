from pathlib import Path

import pytest
from imageio.v2 import imread
from numpy.testing import assert_almost_equal

from conftest import torch_param
from psnr_hvsm._psnr_hvsm import hvs_hvsm_mse_tiles as hvs_hvsm_mse_tiles_cpp
from psnr_hvsm._psnr_hvsm import psnr_hvs_hvsm as psnr_hvs_hvsm_cpp
from psnr_hvsm.numpy import hvs_hvsm_mse_tiles as hvs_hvsm_mse_tiles_np
from psnr_hvsm.numpy import psnr_hvs_hvsm as psnr_hvs_hvsm_np
try:
    from psnr_hvsm.torch import hvs_hvsm_mse_tiles as hvs_hvsm_mse_tiles_pt
    from psnr_hvsm.torch import psnr_hvs_hvsm as psnr_hvs_hvsm_pt
except ModuleNotFoundError as error:
    if error.name not in ('torch', 'torch_dct'):
        raise
    hvs_hvsm_mse_tiles_pt = None
    psnr_hvs_hvsm_pt = None

test_dir = Path(__file__).parent.absolute()
org = imread((test_dir / 'baboon.png').as_posix(), mode='L').astype(float) / 255
tst = imread((test_dir / 'baboon_msk.png').as_posix(), mode='L').astype(float) / 255


@pytest.mark.parametrize('psnr_hvs_hvsm_impl', [
    pytest.param(psnr_hvs_hvsm_cpp, id='cpp'),
    pytest.param(psnr_hvs_hvsm_np, id='numpy'),
    torch_param(psnr_hvs_hvsm_pt),
])
def test_psnr_hvs_hvsm(psnr_hvs_hvsm_impl):
    phvs, phvsm = psnr_hvs_hvsm_impl(org, tst)

    assert_almost_equal(phvs, 34.42705450576435, decimal=12)  # result obtained in octave
    assert_almost_equal(phvsm, 51.64722121999962, decimal=12)  # result obtained in octave


@pytest.mark.parametrize('hvs_hvsm_mse_tiles_impl', [
    pytest.param(hvs_hvsm_mse_tiles_cpp, id='cpp'),
    pytest.param(hvs_hvsm_mse_tiles_np, id='numpy'),
    torch_param(hvs_hvsm_mse_tiles_pt),
])
def test_hvs_hvsm_mse_tiles(hvs_hvsm_mse_tiles_impl):
    hvs_tiles, hvsm_tiles = hvs_hvsm_mse_tiles_impl(org, tst)

    assert_almost_equal(hvs_tiles.mean(), 0.00036082327946204164, decimal=12)
    assert_almost_equal(hvsm_tiles.mean(), 6.843493797885232e-06, decimal=12)
