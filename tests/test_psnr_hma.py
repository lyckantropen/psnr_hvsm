from pathlib import Path

import cv2
import numpy as np
from numpy.testing import assert_almost_equal

from psnr_hvsm import psnr_ha_hma

test_dir = Path(__file__).parent.absolute()
org = cv2.imread((test_dir / 'kodim01.png').as_posix(), cv2.IMREAD_ANYDEPTH).astype(float) / 255
tst = cv2.imread((test_dir / 'kodim01_contrast.png').as_posix(), cv2.IMREAD_ANYDEPTH).astype(float) / 255


def test_psnr_hma():
    psnr_ha, psnr_hma = psnr_ha_hma(org, tst)

    assert_almost_equal(psnr_ha, 17.79500584262193)  # result obtained in octave
    assert_almost_equal(psnr_hma, 18.38369145203613)  # result obtained in octave
