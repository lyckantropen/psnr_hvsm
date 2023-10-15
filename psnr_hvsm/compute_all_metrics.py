"""Define a function for computing PSNR-HVS, PSNR-HVS-M, PSNR-HA, PSNR-HMA and PSNR for a pair of two images."""
from typing import Optional, Tuple

import numpy as np

from . import (backend, bt601ycbcr, bt601ypbpr, psnr, psnr_ha_hma,
               psnr_ha_hma_color, psnr_hvs_hvsm)

if backend == 'torch':
    import torch


def compute_all_metrics(org, dst,
                        max_value: Optional[float] = None,
                        hma_luma: bool = False) -> Tuple[float, float, float, float, float]:
    """Compute PSNR-HVS, PSNR-HVS-M, PSNR-HA, PSNR-HMA and PSNR for a pair of two images."""
    if len(org.shape) > 2:
        # color processing
        if backend in ['numpy', 'cpp']:
            if org.dtype == np.uint8:
                # use BT.601 YCbCr
                xy, xcb, xcr = bt601ycbcr(org)
                yy, ycb, ycr = bt601ycbcr(dst)
                org = org.astype(float) / 255
                dst = dst.astype(float) / 255
            else:
                if max_value is None:
                    raise Exception('Please provide max_value for image files that are >8bpc.')
                org = org.astype(float) / max_value
                dst = dst.astype(float) / max_value
                # assuming analog values (we don't know the ITU rec. to use)
                xy, xcb, xcr = bt601ypbpr(org)
                yy, ycb, ycr = bt601ypbpr(dst)
        elif backend == 'torch':
            if org.dtype == torch.uint8:
                # use BT.601 YCbCr
                xy, xcb, xcr = bt601ycbcr(org)
                yy, ycb, ycr = bt601ycbcr(dst)
                org = org.to(torch.float64) / 255
                dst = dst.to(torch.float64) / 255
            else:
                if max_value is None:
                    raise Exception('Please provide max_value for image files that are >8bpc.')
                org = org.to(torch.float64) / max_value
                dst = dst.to(torch.float64) / max_value
                # assuming analog values (we don't know the ITU rec. to use)
                xy, xcb, xcr = bt601ypbpr(org)
                yy, ycb, ycr = bt601ypbpr(dst)

        if not hma_luma:
            psnr_ha, psnr_hma = psnr_ha_hma_color(xy, xcb, xcr, yy, ycb, ycr)
        else:
            psnr_ha, psnr_hma = psnr_ha_hma(xy, yy)

    else:
        if org.dtype == np.uint8:
            xy = org.astype(float) / 255
            yy = dst.astype(float) / 255
        else:
            if max_value is None:
                raise Exception('Please provide max_value for image files that are >8bpc.')
            xy = org.astype(float) / max_value
            yy = dst.astype(float) / max_value

        psnr_ha, psnr_hma = psnr_ha_hma(xy, yy)

    psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(xy, yy)
    psnr_mse = psnr(org, dst)  # add PSNR for good measure

    if backend == 'torch':
        return psnr_hvs.item(), psnr_hvsm.item(), psnr_ha.item(), psnr_hma.item(), psnr_mse.item()
    else:
        return psnr_hvs, psnr_hvsm, psnr_ha, psnr_hma, psnr_mse
