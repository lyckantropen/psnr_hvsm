"""Implementation of PSNR-HA and PSNR-HMA."""

import numpy as np

from . import hvs_mse, hvsm_mse

COEF1 = 0.002
COEF2 = 0.25
COEF3 = 0.04


def _get_psnr(s, mv):
    if s == 0:
        return 100
    else:
        return min(10.0 * np.log10(mv**2/s), 100.0)


def psnr_ha_hma(x, y):
    """Compute the PSNR-HA and PSNR-HMA metrics for a pair of normalised single-channel images x and y."""
    xm = x.mean()
    delt = xm - y.mean()
    c = y + delt
    cm = c.mean()

    popr = np.sum((x-xm)*(c-cm))/np.sum(np.power(c-cm, 2))

    d = c*popr

    m1 = hvs_mse(x, c).mean()
    m2 = hvs_mse(x, d).mean()
    n1 = hvsm_mse(x, c).mean()
    n2 = hvsm_mse(x, d).mean()

    if m1 > m2:
        if popr < 1:
            m1 = m2 + (m1-m2)*COEF1
        else:
            m1 = m2 + (m1-m2)*COEF2

    m = m1 + delt**2*COEF3

    if n1 > n2:
        if popr < 1:
            n1 = n2 + (n1-n2)*COEF1
        else:
            n1 = n2 + (n1-n2)*COEF2

    n = n1 + delt**2*COEF3

    psnr_ha = _get_psnr(m, 1)
    psnr_hma = _get_psnr(n, 1)

    return psnr_ha, psnr_hma
