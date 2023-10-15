"""Implementation of PSNR-HA and PSNR-HMA."""
from typing import Tuple

import numpy as np

from .psnr import get_psnr
from .psnr_hvsm import hvs_hvsm_mse

COEF1 = 0.002
COEF2 = 0.25
COEF3 = 0.04
COEF4 = 0.5


def ha_hma_mse(x: np.ndarray, y: np.ndarray, hvs_hvsm_mse_impl=hvs_hvsm_mse) -> Tuple[float, float]:
    """Compute the contrast-corrected HVS and HVS-M MSE between two single-channel, normalized images."""
    xm = x.mean()
    delt = xm - y.mean()
    c = y + delt
    cm = c.mean()

    div = np.sum(np.power(c - cm, 2))
    if div == 0.0:
        popr = 1.0
    else:
        popr = np.sum((x - xm) * (c - cm)) / div

    d = cm + (c - cm) * popr

    m1, n1 = hvs_hvsm_mse_impl(x, c)
    m2, n2 = hvs_hvsm_mse_impl(x, d)

    if m1 > m2:
        if popr < 1:
            m1 = m2 + (m1 - m2) * COEF1
        else:
            m1 = m2 + (m1 - m2) * COEF2

    m = m1 + (delt**2) * COEF3

    if n1 > n2:
        if popr < 1:
            n1 = n2 + (n1 - n2) * COEF1
        else:
            n1 = n2 + (n1 - n2) * COEF2

    n = n1 + (delt**2) * COEF3

    return m, n


def psnr_ha_hma(x: np.ndarray, y: np.ndarray, hvs_hvsm_mse_impl=hvs_hvsm_mse) -> Tuple[float, float]:
    """Compute the PSNR-HA and PSNR-HMA metrics for a pair of normalized single-channel images x and y."""
    m, n = ha_hma_mse(x, y, hvs_hvsm_mse_impl)

    psnr_ha = get_psnr(m, 1)
    psnr_hma = get_psnr(n, 1)

    return psnr_ha, psnr_hma


def psnr_ha_hma_color(xy: np.ndarray, xcb: np.ndarray, xcr: np.ndarray, yy: np.ndarray, ycb: np.ndarray,
                      ycr: np.ndarray, hvs_hvsm_mse_impl=hvs_hvsm_mse) -> Tuple[float, float]:
    """Compute the PSNR-HA and PSNR-HMA metrics between two RGB normalized images."""
    my, ny = ha_hma_mse(xy, yy, hvs_hvsm_mse_impl)
    mcb, ncb = ha_hma_mse(xcb, ycb, hvs_hvsm_mse_impl)
    mcr, ncr = ha_hma_mse(xcr, ycr, hvs_hvsm_mse_impl)

    m = (my + COEF4 * (mcb + mcr)) / 2
    n = (ny + COEF4 * (ncb + ncr)) / 2

    psnr_ha = get_psnr(m, 1)
    psnr_hma = get_psnr(n, 1)

    return psnr_ha, psnr_hma
