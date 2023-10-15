"""Implementation of PSNR-HA and PSNR-HMA."""
from typing import Tuple

import numpy as np
import torch

from .psnr import get_psnr
from .psnr_hvsm import hvs_hvsm_mse

COEF1 = 0.002
COEF2 = 0.25
COEF3 = 0.04
COEF4 = 0.5


def ha_hma_mse(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the contrast-corrected HVS and HVS-M MSE between two single-channel, normalized images."""
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    xm = x.mean()
    delt = xm - y.mean()
    c = y + delt
    cm = c.mean()

    div = torch.sum(torch.pow(c - cm, 2))
    if div == 0.0:
        popr = torch.tensor(1.0, device=x.device)
    else:
        popr = torch.sum((x - xm) * (c - cm)) / div

    d = cm + (c - cm) * popr

    m1, n1 = hvs_hvsm_mse(x, c)
    m2, n2 = hvs_hvsm_mse(x, d)

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


def psnr_ha_hma(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the PSNR-HA and PSNR-HMA metrics for a pair of normalized single-channel images x and y."""
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    m, n = ha_hma_mse(x, y)

    psnr_ha = get_psnr(m, 1)
    psnr_hma = get_psnr(n, 1)

    return psnr_ha, psnr_hma


def psnr_ha_hma_color(xy: torch.Tensor,
                      xcb: torch.Tensor,
                      xcr: torch.Tensor,
                      yy: torch.Tensor,
                      ycb: torch.Tensor,
                      ycr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the PSNR-HA and PSNR-HMA metrics between two RGB normalized images."""
    if isinstance(xy, np.ndarray):
        xy = torch.tensor(xy)
    if isinstance(xcb, np.ndarray):
        xcb = torch.tensor(xcb)
    if isinstance(xcr, np.ndarray):
        xcr = torch.tensor(xcr)
    if isinstance(yy, np.ndarray):
        yy = torch.tensor(yy)
    if isinstance(ycb, np.ndarray):
        ycb = torch.tensor(ycb)
    if isinstance(ycr, np.ndarray):
        ycr = torch.tensor(ycr)

    my, ny = ha_hma_mse(xy, yy)
    mcb, ncb = ha_hma_mse(xcb, ycb)
    mcr, ncr = ha_hma_mse(xcr, ycr)

    m = (my + COEF4 * (mcb + mcr)) / 2
    n = (ny + COEF4 * (ncb + ncr)) / 2

    psnr_ha = get_psnr(m, 1)
    psnr_hma = get_psnr(n, 1)

    return psnr_ha, psnr_hma
