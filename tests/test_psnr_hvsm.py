from pathlib import Path

import cv2
import numpy as np
from numpy.testing import assert_almost_equal

from psnr_hvsm import hvs_mse, hvsm_mse, psnr_hvs, psnr_hvsm

test_dir = Path(__file__).parent.absolute()
org = cv2.imread((test_dir / 'baboon.png').as_posix(), cv2.IMREAD_GRAYSCALE).astype(float) / 255
tst = cv2.imread((test_dir / 'baboon_msk.png').as_posix(), cv2.IMREAD_GRAYSCALE).astype(float) / 255


def _get_psnr(s, mv):
    if s == 0:
        return 100
    else:
        return min(10.0 * np.log10(mv**2/s), 100.0)


def test_psnr_hvs_and_hvs_mse_on_baboon_image():
    mse_hvs = hvs_mse(org, tst)
    phvs = psnr_hvs(org, tst)

    assert_almost_equal(phvs, 34.427054505764424)
    assert_almost_equal(phvs, _get_psnr(mse_hvs.mean(), 1))


def test_psnr_hvsm_and_hvms_mse_on_baboon_image():
    mse_hvsm = hvsm_mse(org, tst)
    phvsm = psnr_hvsm(org, tst)

    assert_almost_equal(phvsm, 51.647221219999615)
    assert_almost_equal(phvsm, _get_psnr(mse_hvsm.mean(), 1))
