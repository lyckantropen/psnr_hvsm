from pathlib import Path

import cv2
import numpy as np

from psnr_hvsm import hvs_mse, psnr_hvs, psnr_hvsm
from psnr_hvsm.bt601 import bt601luma

refs = sorted(list(Path('d:\\Devel\\tid2013\\reference_images').glob('*.bmp')))


def _get_psnr(s, mv):
    if s == 0:
        return 100
    else:
        return min(10.0 * np.log10(mv**2/s), 100.0)


for ref in refs:
    ref_img = cv2.imread(ref.as_posix(), cv2.IMREAD_COLOR)
    ref_img = bt601luma(ref_img)
    for d in range(1, 25):
        for i in range(1, 6):
            tst = Path('d:\\Devel\\tid2013\\distorted_images') / f'{ref.stem}_{d:02}_{i}.bmp'
            tst_img = cv2.imread(tst.as_posix(), cv2.IMREAD_COLOR)
            tst_img = bt601luma(tst_img)

            print(f'{psnr_hvs(ref_img, tst_img):.4f}')
