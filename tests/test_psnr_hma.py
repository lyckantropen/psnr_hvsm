from pathlib import Path

from imageio import imread
from numpy.testing import assert_almost_equal

from psnr_hvsm import ha_hma_mse, psnr_ha_hma
from psnr_hvsm.bt601 import bt601ycbcr

test_dir = Path(__file__).parent.absolute()


def test_psnr_hma():
    org = imread((test_dir / 'kodim01.png').as_posix()).astype(float) / 255
    tst = imread((test_dir / 'kodim01_contrast.png').as_posix()).astype(float) / 255

    psnr_ha, psnr_hma = psnr_ha_hma(org, tst)

    assert_almost_equal(psnr_ha, 23.72516252033972)  # result obtained in octave
    assert_almost_equal(psnr_hma, 24.30979923387402, decimal=4)  # result obtained in octave


def test_ha_hma_mse():
    org = imread((test_dir / 'kodim01.png').as_posix()).astype(float)
    tst = imread((test_dir / 'kodim01_contrast.png').as_posix()).astype(float)

    mse_ha, mse_hma = ha_hma_mse(org, tst)

    assert_almost_equal(mse_ha, 275.7808519088271)  # result obtained in octave
    assert_almost_equal(mse_hma, 241.0462821953413, decimal=3)  # result obtained in octave


def test_ha_hma_mse_color():
    org = imread((test_dir / 'kodim01_c.png').as_posix()).astype(float)
    tst = imread((test_dir / 'kodim01_c_contrast.png').as_posix()).astype(float)[:, :, :3]

    y0, _, _ = bt601ycbcr(org)
    y1, _, _ = bt601ycbcr(tst)

    mse_ha, mse_hma = ha_hma_mse(y0 * 255, y1 * 255)

    assert_almost_equal(mse_ha, 314.1129805328927)  # result obtained in octave
    assert_almost_equal(mse_hma, 278.3264049014794, decimal=2)  # result obtained in octave
